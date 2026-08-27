#!/usr/bin/env python3
"""Run the local, reviewed Attune cascade on one or more WAV files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from attune.baselines.adapters import (
    BaselineInput,
    BaselinePrediction,
    build_partial_output,
)
from attune.baselines.cascade import AttuneCascade
from attune.evaluation.report import RuntimeMetrics
from attune.inference.timeline import render_demo_html
from attune.models.temporal_probe import (
    dcase_checkpoint_acceptance,
    read_temporal_checkpoint,
)
from attune.schema.output import AffectCategory, AttuneOutput
from attune.schema.xml import render_xml

SENSEVOICE_CANDIDATES = (
    Path("artifacts/starss23-scene-raster/sensevoice-small"),
    Path("data/raw/model-cache/sensevoice-small"),
)
EMOTION2VEC_CANDIDATES = (
    Path("data/raw/model-cache/emotion2vec-plus"),
    Path("data/raw/model-cache/emotion2vec_plus_base"),
)
VOCALSOUND_CANDIDATES = (Path("artifacts/event-probe/head.pt"),)
FSD50K_CANDIDATES = (Path("artifacts/fsd50k-event-probe/head.pt"),)
DCASE_CANDIDATES = (
    Path("artifacts/dcase-frame-localization/frame-head.pt"),
    Path("artifacts/dcase-localization/frame-head.pt"),
    Path("artifacts/dcase-localization/temporal-head.pt"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+", help="local PCM WAV file(s)")
    parser.add_argument("--output", type=Path, help="JSON file, or directory for multiple WAVs")
    parser.add_argument(
        "--xml-output",
        help="optional XML file/directory; use '-' for stdout (requires --output)",
    )
    parser.add_argument(
        "--html-output",
        help="optional playable HTML timeline; '-' is stdout and requires --output",
    )
    parser.add_argument(
        "--sensevoice-path",
        type=Path,
        default=_path_from_env("ATTUNE_SENSEVOICE_SMALL_PATH"),
    )
    parser.add_argument(
        "--emotion2vec-path",
        type=Path,
        default=_path_from_env("ATTUNE_EMOTION2VEC_PLUS_PATH"),
    )
    parser.add_argument(
        "--vocalsound-probe",
        type=Path,
        default=_path_from_env("ATTUNE_VOCALSOUND_PROBE_PATH"),
    )
    parser.add_argument(
        "--fsd50k-probe",
        type=Path,
        default=_path_from_env("ATTUNE_FSD50K_PROBE_PATH"),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/cascade-sensevoice-embeddings"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("configs/calibration/phase2.json"),
    )
    parser.add_argument(
        "--temporal-head",
        type=Path,
        default=_path_from_env("ATTUNE_TEMPORAL_HEAD_PATH"),
        help="optional gated DCASE frame-head; omitted when no local DCASE checkpoint exists",
    )
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="schema/CLI test path only; emits an explicit placeholder without loading models",
    )
    return parser.parse_args(argv)


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _first_existing(candidates: Sequence[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _discover_required_artifacts(arguments: argparse.Namespace) -> None:
    if arguments.sensevoice_path is None:
        arguments.sensevoice_path = _first_existing(SENSEVOICE_CANDIDATES)
    if arguments.emotion2vec_path is None:
        arguments.emotion2vec_path = _first_existing(EMOTION2VEC_CANDIDATES)
    if arguments.vocalsound_probe is None:
        arguments.vocalsound_probe = _first_existing(VOCALSOUND_CANDIDATES)
    if arguments.fsd50k_probe is None:
        arguments.fsd50k_probe = _first_existing(FSD50K_CANDIDATES)


def _resolve_dcase_temporal_head(path: Path | None, *, explicit: bool) -> Path | None:
    """Return one DCASE gated head, or None. Never loads STARSS23 checkpoints."""
    if path is None:
        for candidate in DCASE_CANDIDATES:
            if not candidate.is_file():
                continue
            try:
                accepted, _reason = dcase_checkpoint_acceptance(read_temporal_checkpoint(candidate))
            except (OSError, RuntimeError, ValueError):
                continue
            if accepted:
                return candidate
        return None
    if not path.is_file():
        raise ValueError(f"temporal head is missing: {path}")
    accepted, reason = dcase_checkpoint_acceptance(read_temporal_checkpoint(path))
    if accepted:
        return path
    message = reason or "temporal head is not a gated DCASE checkpoint"
    if explicit:
        raise ValueError(message)
    return None


def _require_real_artifacts(arguments: argparse.Namespace) -> None:
    if os.environ.get("ATTUNE_SENSEVOICE_LICENSE_REVIEWED") != "1":
        raise ValueError(
            "review the SenseVoice model licence, then set ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1"
        )
    _discover_required_artifacts(arguments)
    required = {
        "SenseVoice-Small weights (--sensevoice-path or ATTUNE_SENSEVOICE_SMALL_PATH)": (
            arguments.sensevoice_path
        ),
        "emotion2vec+ weights (--emotion2vec-path or ATTUNE_EMOTION2VEC_PLUS_PATH)": (
            arguments.emotion2vec_path
        ),
        "VocalSound probe head (--vocalsound-probe or ATTUNE_VOCALSOUND_PROBE_PATH)": (
            arguments.vocalsound_probe
        ),
        "FSD50K probe head (--fsd50k-probe or ATTUNE_FSD50K_PROBE_PATH)": (arguments.fsd50k_probe),
        "Phase 2 calibration": arguments.calibration,
    }
    missing = [
        f"{name}: {path if path is not None else 'path not configured'}"
        for name, path in required.items()
        if path is None or not path.exists()
    ]
    if missing:
        raise ValueError(
            "required local artifacts are missing; downloads are disabled:\n  - "
            + "\n  - ".join(missing)
        )


def _fixture_prediction(audio_path: Path) -> BaselinePrediction:
    distribution = {label: 0.1 for label in AffectCategory}
    distribution[AffectCategory.NEUTRAL] = 0.3
    output = build_partial_output(
        BaselineInput(audio_path=audio_path),
        model_name="attune-cli-fixture-placeholder",
        transcript="",
        category=AffectCategory.NEUTRAL,
        distribution=distribution,
        abstain=True,
    )
    return BaselinePrediction(
        output=output,
        runtime=RuntimeMetrics.measured(
            audio_seconds=output.audio.duration_ms / 1000,
            elapsed_seconds=0.0,
        ),
        diagnostics={
            "fixture_mode": True,
            "note": "Schema/CLI wiring only; no model inference or scientific prediction.",
            "dcase_head_configured": False,
        },
    )


def _build_cascade(arguments: argparse.Namespace, temporal_head: Path | None) -> AttuneCascade:
    _require_real_artifacts(arguments)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MODELSCOPE_OFFLINE": "1",
            "FUNASR_DISABLE_UPDATE": "1",
        }
    )
    cascade = AttuneCascade(
        sensevoice_checkpoint=arguments.sensevoice_path,
        emotion2vec_checkpoint=arguments.emotion2vec_path,
        vocalsound_probe_checkpoint=arguments.vocalsound_probe,
        fsd50k_probe_checkpoint=arguments.fsd50k_probe,
        embedding_cache=arguments.embedding_cache,
        calibration_path=arguments.calibration,
        temporal_head_checkpoints=(temporal_head,) if temporal_head is not None else (),
    )
    available, reason = cascade.availability()
    if not available:
        raise ValueError(reason or "Attune cascade is unavailable")
    return cascade


def _destination_for(audio_path: Path, destination: Path, suffix: str) -> Path:
    return destination / f"{audio_path.stem}.attune{suffix}"


def _write_json(outputs: list[tuple[Path, AttuneOutput]], destination: Path | None) -> None:
    if destination is None:
        if len(outputs) == 1:
            print(json.dumps(outputs[0][1].model_dump(mode="json"), indent=2))
        else:
            for _, output in outputs:
                print(json.dumps(output.model_dump(mode="json"), separators=(",", ":")))
        return
    if len(outputs) == 1:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(outputs[0][1].model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        return
    destination.mkdir(parents=True, exist_ok=True)
    for audio_path, output in outputs:
        path = _destination_for(audio_path, destination, ".json")
        path.write_text(
            json.dumps(output.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )


def _write_xml(outputs: list[tuple[Path, AttuneOutput]], destination: str | None) -> None:
    if destination is None:
        return
    if destination == "-":
        for index, (_, output) in enumerate(outputs):
            if index:
                print()
            print(render_xml(output))
        return
    path = Path(destination)
    if len(outputs) == 1:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_xml(outputs[0][1]) + "\n", encoding="utf-8")
        return
    path.mkdir(parents=True, exist_ok=True)
    for audio_path, output in outputs:
        _destination_for(audio_path, path, ".xml").write_text(
            render_xml(output) + "\n",
            encoding="utf-8",
        )


def _audio_src_for(audio_path: Path, html_path: Path) -> str:
    try:
        return os.path.relpath(audio_path.resolve(), html_path.parent.resolve())
    except ValueError:
        return str(audio_path)


def _write_html(
    outputs: list[tuple[Path, AttuneOutput]],
    destination: str | None,
    *,
    dcase_head_configured: bool,
    fixture: bool,
) -> None:
    if destination is None:
        return

    def page(audio_path: Path, output: AttuneOutput, html_path: Path | None) -> str:
        audio_src = str(audio_path) if html_path is None else _audio_src_for(audio_path, html_path)
        return render_demo_html(
            output,
            audio_src=audio_src,
            dcase_head_configured=dcase_head_configured,
            fixture=fixture,
        )

    if destination == "-":
        for index, (audio_path, output) in enumerate(outputs):
            if index:
                print()
            print(page(audio_path, output, None))
        return
    path = Path(destination)
    if len(outputs) == 1:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page(outputs[0][0], outputs[0][1], path) + "\n", encoding="utf-8")
        return
    path.mkdir(parents=True, exist_ok=True)
    for audio_path, output in outputs:
        html_path = _destination_for(audio_path, path, ".html")
        html_path.write_text(page(audio_path, output, html_path) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    missing_audio = [path for path in arguments.audio if not path.is_file()]
    if missing_audio:
        print(f"error: WAV file does not exist: {missing_audio[0]}", file=sys.stderr)
        return 2
    if arguments.xml_output == "-" and arguments.output is None:
        print("error: XML stdout requires --output for authoritative JSON", file=sys.stderr)
        return 2
    if arguments.html_output == "-" and arguments.output is None:
        print("error: HTML stdout requires --output for authoritative JSON", file=sys.stderr)
        return 2
    try:
        explicit_temporal = arguments.temporal_head is not None
        temporal_head = None
        if not arguments.fixture_mode:
            temporal_head = _resolve_dcase_temporal_head(
                arguments.temporal_head,
                explicit=explicit_temporal,
            )
        cascade = None if arguments.fixture_mode else _build_cascade(arguments, temporal_head)
        outputs = []
        for audio_path in arguments.audio:
            prediction = (
                _fixture_prediction(audio_path)
                if cascade is None
                else cascade.predict(BaselineInput(audio_path=audio_path))
            )
            outputs.append((audio_path, prediction.output))
        _write_json(outputs, arguments.output)
        _write_xml(outputs, arguments.xml_output)
        _write_html(
            outputs,
            arguments.html_output,
            dcase_head_configured=temporal_head is not None,
            fixture=arguments.fixture_mode,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
