#!/usr/bin/env python3
"""Prepare deterministic BERSt gain variants for post-selection robustness tests."""

from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path

import numpy as np

from attune.integrity import file_digest
from attune.training.prepare import SourceRow, load_source_rows

DEFAULT_GAINS_DB = (-12.0, -6.0, 0.0, 6.0, 12.0)


def gain_label(gain_db: float) -> str:
    sign = "m" if gain_db < 0 else "p" if gain_db > 0 else "z"
    magnitude = f"{abs(gain_db):g}".replace(".", "p")
    return f"{sign}{magnitude}db"


def apply_gain(source: Path, target: Path, gain_db: float) -> dict[str, float | int]:
    with wave.open(str(source), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        frame_count = audio.getnframes()
        frames = audio.readframes(frame_count)
    if channels != 1 or sample_rate != 16_000 or sample_width != 2:
        raise ValueError(f"{source}: expected 16 kHz mono PCM16 WAV")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    scaled = np.rint(samples * 10 ** (gain_db / 20.0))
    clipped = (scaled < -32_768) | (scaled > 32_767)
    output = np.clip(scaled, -32_768, 32_767).astype("<i2")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part.wav")
    with wave.open(str(temporary), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(output.tobytes())
    temporary.replace(target)
    return {
        "gain_db": gain_db,
        "sample_count": len(output),
        "clipped_samples": int(clipped.sum()),
        "clipping_ratio": float(clipped.mean()) if len(clipped) else 0.0,
    }


def prepare_gain_sweep(
    source_manifest: Path,
    output_manifest: Path,
    audio_dir: Path,
    *,
    gains_db: tuple[float, ...] = DEFAULT_GAINS_DB,
    split: str = "sealed_test",
) -> tuple[list[SourceRow], dict]:
    source_rows = [row for row in load_source_rows(source_manifest) if row.split == split]
    if not source_rows:
        raise ValueError(f"source manifest contains no {split!r} rows")
    if len(gains_db) != len(set(gains_db)):
        raise ValueError("gain values must be unique")

    prepared: list[SourceRow] = []
    gain_reports = []
    for gain_db in gains_db:
        label = gain_label(gain_db)
        clipped_samples = sample_count = 0
        maximum_clipping_ratio = 0.0
        for row in source_rows:
            target = audio_dir / label / f"{row.clip_id}.wav"
            audio_report = apply_gain(row.audio_path, target, gain_db)
            clipped_samples += int(audio_report["clipped_samples"])
            sample_count += int(audio_report["sample_count"])
            maximum_clipping_ratio = max(
                maximum_clipping_ratio, float(audio_report["clipping_ratio"])
            )
            prepared.append(
                row.model_copy(
                    update={
                        "clip_id": f"{row.clip_id}-gain-{label}",
                        "dataset_id": f"berst_v1_gain_{label}",
                        "audio_path": Path(
                            os.path.relpath(target.resolve(), output_manifest.parent.resolve())
                        ),
                        "audio_sha256": file_digest(target),
                    }
                )
            )
        gain_reports.append(
            {
                "gain_db": gain_db,
                "dataset_id": f"berst_v1_gain_{label}",
                "clips": len(source_rows),
                "clipped_samples": clipped_samples,
                "sample_count": sample_count,
                "clipping_ratio": clipped_samples / sample_count if sample_count else 0.0,
                "maximum_clip_clipping_ratio": maximum_clipping_ratio,
            }
        )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text("".join(row.model_dump_json() + "\n" for row in prepared))
    report = {
        "schema_version": "1.0",
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": file_digest(source_manifest),
        },
        "split": split,
        "source_clips": len(source_rows),
        "prepared_clips": len(prepared),
        "gains": gain_reports,
        "output_manifest": {
            "path": str(output_manifest),
            "sha256": file_digest(output_manifest),
        },
    }
    return prepared, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/manifests/berst-v0.1-source.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manifests/berst-gain-sweep-v0.1-source.jsonl"),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("artifacts/data/berst/gain-sweep-v0.1/audio"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/data/berst/gain-sweep-v0.1/preparation.json"),
    )
    parser.add_argument("--gain-db", type=float, action="append")
    parser.add_argument("--split", default="sealed_test")
    arguments = parser.parse_args()
    _rows, report = prepare_gain_sweep(
        arguments.source,
        arguments.output,
        arguments.audio_dir,
        gains_db=tuple(arguments.gain_db or DEFAULT_GAINS_DB),
        split=arguments.split,
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"prepared_clips": report["prepared_clips"]}))


if __name__ == "__main__":
    main()
