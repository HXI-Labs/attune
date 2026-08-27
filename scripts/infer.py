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
from attune.schema.output import AffectCategory, AttuneOutput
from attune.schema.xml import render_xml


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+", help="local PCM WAV file(s)")
    parser.add_argument("--output", type=Path, help="JSON file, or directory for multiple WAVs")
    parser.add_argument(
        "--xml-output",
        help="optional XML file/directory; use '-' for stdout (requires --output)",
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
        help="optional held-out-gated DCASE frame head for laugh/cough/throat-clear spans",
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


def _require_real_artifacts(arguments: argparse.Namespace) -> None:
    if os.environ.get("ATTUNE_SENSEVOICE_LICENSE_REVIEWED") != "1":
        raise ValueError(
            "review the SenseVoice model licence, then set ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1"
        )
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
        },
    )


def _build_cascade(arguments: argparse.Namespace) -> AttuneCascade:
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
        temporal_head_checkpoint=arguments.temporal_head,
    )
    available, reason = cascade.availability()
    if not available:
        raise ValueError(reason or "Attune cascade is unavailable")
    return cascade


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
        path = destination / f"{audio_path.stem}.attune.json"
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
        (path / f"{audio_path.stem}.attune.xml").write_text(
            render_xml(output) + "\n",
            encoding="utf-8",
        )


def run(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    missing_audio = [path for path in arguments.audio if not path.is_file()]
    if missing_audio:
        print(f"error: WAV file does not exist: {missing_audio[0]}", file=sys.stderr)
        return 2
    if arguments.xml_output == "-" and arguments.output is None:
        print("error: XML stdout requires --output for authoritative JSON", file=sys.stderr)
        return 2
    try:
        cascade = None if arguments.fixture_mode else _build_cascade(arguments)
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
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
