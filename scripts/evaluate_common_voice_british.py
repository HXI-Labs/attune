#!/usr/bin/env python3
"""Run offline Whisper-Small and SenseVoiceSmall on the British CC0 slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prepare_common_voice_british import LICENCE, load_manifest, safe_target, verify

from attune.baselines.adapters import BaselineInput, SenseVoiceSmallAdapter, WhisperSmallAdapter
from attune.evaluation.metrics import corpus_character_error_rate, corpus_word_error_rate

COMPARATORS = {
    "crema_d_acted_us": {"sensevoice-small": 0.0806, "whisper-small": 0.1226},
    "ghanaian_english_nc": {"sensevoice-small": 0.2365, "whisper-small": 0.2562},
}
MODEL_METADATA = {
    "whisper-small": {
        "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
        "attribution": "OpenAI Whisper-Small; upstream Whisper MIT licence.",
    },
    "sensevoice-small": {
        "revision": "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
        "attribution": (
            "SenseVoiceSmall by FunASR/FunAudioLLM; "
            "FunASR Model Open Source License Agreement v1.1."
        ),
    },
}


def normalize_asr(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def checkpoint_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".bin", ".pt", ".safetensors"}:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(path.relative_to(root))] = digest.hexdigest()
    return hashes


def _asr_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    return {
        "wer": corpus_word_error_rate(references, hypotheses),
        "cer": corpus_character_error_rate(references, hypotheses),
    }


def evaluate_runner(
    runner: WhisperSmallAdapter | SenseVoiceSmallAdapter,
    rows: list[dict[str, Any]],
    cache_root: Path,
) -> dict[str, Any]:
    references: list[str] = []
    hypotheses: list[str] = []
    accent_outputs: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    failures: list[dict[str, str]] = []
    elapsed_s = 0.0
    audio_s = 0.0
    for index, row in enumerate(rows, 1):
        try:
            prediction = runner.predict(
                BaselineInput(
                    audio_path=safe_target(cache_root, row["cache_path"]),
                    language_hint="en",
                )
            )
        except Exception as error:
            failures.append(
                {
                    "clip_id": row["clip_id"],
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            continue
        reference = normalize_asr(row["transcript"])
        hypothesis = normalize_asr(prediction.output.transcript.text)
        references.append(reference)
        hypotheses.append(hypothesis)
        accent_references, accent_hypotheses = accent_outputs[row["accent_value"]]
        accent_references.append(reference)
        accent_hypotheses.append(hypothesis)
        elapsed_s += prediction.runtime.elapsed_seconds
        audio_s += prediction.runtime.audio_seconds
        print(f"{runner.name}: {index}/{len(rows)}", flush=True)
    status = "completed" if not failures else ("partial_failure" if hypotheses else "failed")
    metrics = _asr_metrics(references, hypotheses) if references else {"wer": None, "cer": None}
    wer = metrics["wer"]
    comparisons = {
        name: {
            "reference_wer": values[runner.name],
            "absolute_wer_difference": wer - values[runner.name] if wer is not None else None,
        }
        for name, values in COMPARATORS.items()
    }
    return {
        "runner": runner.name,
        "status": status,
        "evaluated_clips": len(references),
        "failed_clips": len(failures),
        "metrics": {
            **metrics,
            "normalization": "lowercase ASCII alphanumeric tokens; punctuation removed",
            "by_accent": {
                accent: {
                    "clips": len(outputs[0]),
                    **_asr_metrics(*outputs),
                }
                for accent, outputs in sorted(accent_outputs.items())
            },
        },
        "comparisons": comparisons,
        "runtime": {
            "audio_seconds": audio_s,
            "elapsed_seconds": elapsed_s,
            "real_time_factor": elapsed_s / audio_s if audio_s else None,
            "device": "cpu",
        },
        "model": MODEL_METADATA[runner.name],
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/common-voice-british-wer.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/common-voice-british-wer"),
    )
    parser.add_argument("--whisper-path", type=Path, required=True)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/common-voice-british-wer-results.json"),
    )
    arguments = parser.parse_args()

    verify(arguments.manifest, arguments.cache_dir)
    rows = load_manifest(arguments.manifest)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MODELSCOPE_OFFLINE": "1",
            "FUNASR_DISABLE_UPDATE": "1",
            "ATTUNE_SENSEVOICE_LICENSE_REVIEWED": "1",
        }
    )
    started = time.time()
    runners = [
        WhisperSmallAdapter(checkpoint=arguments.whisper_path),
        SenseVoiceSmallAdapter(checkpoint=arguments.sensevoice_path),
    ]
    results = [evaluate_runner(runner, rows, arguments.cache_dir) for runner in runners]
    payload = {
        "report_version": "1",
        "title": "British-English ASR slice — Mozilla Common Voice CC0",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": {
            "dataset": rows[0]["source_dataset"],
            "transport_mirror": rows[0]["source_mirror"],
            "dataset_revision": rows[0]["dataset_revision"],
            "licence": LICENCE,
            "attribution": rows[0]["attribution"],
            "manifest": str(arguments.manifest),
            "clip_count": len(rows),
            "duration_seconds": sum(float(row["duration_s"]) for row in rows),
            "sample_rate_hz": 16_000,
            "channels": 1,
            "speaker_count": len({row["client_id"] for row in rows}),
            "speaker_disjoint": True,
            "accent_field": "accent",
            "accent_values": sorted({row["accent_value"] for row in rows}),
            "accent_counts": {
                accent: sum(row["accent_value"] == accent for row in rows)
                for accent in sorted({row["accent_value"] for row in rows})
            },
            "fine_tuning_performed": False,
            "gate_decision": "not_evaluated_gate_remains_closed",
        },
        "execution": {
            "offline_after_fetch": True,
            "network_disabled_by_model_runtime_flags": True,
            "run_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "wall_seconds": time.time() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "checkpoint_hashes": {
            "whisper-small": checkpoint_hashes(arguments.whisper_path),
            "sensevoice-small": checkpoint_hashes(arguments.sensevoice_path),
        },
        "results": results,
        "attribution": {
            "dataset": "Mozilla Common Voice Corpus 17.0 English; CC0-1.0.",
            "whisper-small": MODEL_METADATA["whisper-small"]["attribution"],
            "sensevoice-small": MODEL_METADATA["sensevoice-small"]["attribution"],
        },
        "limitations": [
            "Accent values are self-declared and do not verify nationality or residence.",
            "The 100-speaker deterministic slice is not population-representative.",
            "CREMA-D is acted US English and Ghanaian English NC is broadcast-domain context; "
            "neither is a matched control.",
            "This WER/CER slice does not evaluate calibration, affect, events, or localization.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote British Common Voice results to {arguments.output}")


if __name__ == "__main__":
    main()
