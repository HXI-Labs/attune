#!/usr/bin/env python3
"""Run the inspected Attune cascade on one local WAV file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

from attune.baselines.adapters import (
    BaselineInput,
    Emotion2VecPlusAdapter,
    SenseVoiceSmallAdapter,
)
from attune.baselines.cascade import ModularCascade
from attune.models.probe_inference import (
    FSD50K_LABEL_MAPPING,
    VOCALSOUND_LABEL_MAPPING,
    FrozenEncoderProvider,
    FrozenLinearProbeHead,
)
from attune.schema.xml import render_xml

DEFAULT_MODEL_ROOT = Path("data/raw/model-cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="local WAV file")
    parser.add_argument(
        "--sensevoice-path",
        type=Path,
        default=_path_from_env(
            "ATTUNE_SENSEVOICE_SMALL_PATH",
            DEFAULT_MODEL_ROOT / "sensevoice-small",
        ),
    )
    parser.add_argument(
        "--emotion2vec-path",
        type=Path,
        default=_path_from_env(
            "ATTUNE_EMOTION2VEC_PLUS_PATH",
            DEFAULT_MODEL_ROOT / "emotion2vec-plus",
        ),
    )
    parser.add_argument(
        "--vocalsound-probe",
        type=Path,
        default=Path("artifacts/event-probe/head.pt"),
    )
    parser.add_argument(
        "--fsd50k-probe",
        type=Path,
        default=Path("artifacts/fsd50k-event-probe/head.pt"),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/cascade-sensevoice-embeddings"),
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="print deterministic XML instead of JSON",
    )
    parser.add_argument(
        "--xml-output",
        type=Path,
        help="also write deterministic XML to this path while printing JSON",
    )
    return parser.parse_args()


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def available_probe_heads(
    *,
    encoder: FrozenEncoderProvider,
    vocalsound_checkpoint: Path,
    fsd50k_checkpoint: Path,
) -> tuple[tuple[FrozenLinearProbeHead, ...], tuple[str, ...]]:
    """Build configured heads and explicitly report missing local checkpoints."""
    configured = (
        (
            "VocalSound",
            "vocalsound-frozen-linear-probe",
            vocalsound_checkpoint,
            VOCALSOUND_LABEL_MAPPING,
        ),
        (
            "FSD50K",
            "fsd50k-frozen-linear-probe",
            fsd50k_checkpoint,
            FSD50K_LABEL_MAPPING,
        ),
    )
    heads = []
    skipped = []
    for source, name, checkpoint, mapping in configured:
        if not checkpoint.is_file():
            skipped.append(f"{source} probe: missing {checkpoint}")
            continue
        heads.append(
            FrozenLinearProbeHead(
                name=name,
                checkpoint=checkpoint,
                encoder=encoder,
                label_mapping=mapping,
            )
        )
    return tuple(heads), tuple(skipped)


def main() -> None:
    arguments = parse_args()
    if not arguments.audio.is_file():
        raise SystemExit(f"error: audio file does not exist: {arguments.audio}")
    if os.environ.get("ATTUNE_SENSEVOICE_LICENSE_REVIEWED") != "1":
        raise SystemExit(
            "error: review the SenseVoice model licence, then set "
            "ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1"
        )
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MODELSCOPE_OFFLINE": "1",
            "FUNASR_DISABLE_UPDATE": "1",
        }
    )
    encoder = FrozenEncoderProvider(arguments.sensevoice_path, arguments.embedding_cache)
    heads, skipped = available_probe_heads(
        encoder=encoder,
        vocalsound_checkpoint=arguments.vocalsound_probe,
        fsd50k_checkpoint=arguments.fsd50k_probe,
    )
    for reason in skipped:
        print(f"Skipped {reason}; continuing with SenseVoice AED + affect.", file=sys.stderr)
    cascade = ModularCascade(
        asr=SenseVoiceSmallAdapter(checkpoint=arguments.sensevoice_path),
        affect=Emotion2VecPlusAdapter(checkpoint=arguments.emotion2vec_path),
        event_heads=heads,
    )
    available, reason = cascade.availability()
    if not available:
        raise SystemExit(f"error: {reason}")
    # FunASR prints model-loading notices and progress bars to stdout. Keep stdout
    # machine-readable by routing dependency chatter to stderr.
    with redirect_stdout(sys.stderr):
        prediction = cascade.predict(BaselineInput(audio_path=arguments.audio))
    if arguments.xml:
        print(render_xml(prediction.output))
        return
    print(json.dumps(prediction.output.model_dump(mode="json"), indent=2))
    if arguments.xml_output is not None:
        arguments.xml_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.xml_output.write_text(render_xml(prediction.output) + "\n", encoding="utf-8")
        print(f"Wrote XML to {arguments.xml_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
