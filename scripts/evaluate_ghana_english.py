#!/usr/bin/env python3
"""Run offline ASR WER/CER on the NC research-only Ghanaian-English slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attune.baselines.adapters import BaselineInput, SenseVoiceSmallAdapter, WhisperSmallAdapter
from attune.evaluation.metrics import corpus_character_error_rate, corpus_word_error_rate

from prepare_ghana_english import LICENCE, load_manifest, safe_target, verify

NC_LABEL = "NC research-only"
CREMA_D_WER = {"sensevoice-small": 0.0806, "whisper-small": 0.1226}
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


def evaluate_runner(
    runner: WhisperSmallAdapter | SenseVoiceSmallAdapter,
    rows: list[dict[str, Any]],
    cache_root: Path,
) -> dict[str, Any]:
    references: list[str] = []
    hypotheses: list[str] = []
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
                    "licence_scope": NC_LABEL,
                }
            )
            continue
        references.append(normalize_asr(row["transcript"]))
        hypotheses.append(normalize_asr(prediction.output.transcript.text))
        elapsed_s += prediction.runtime.elapsed_seconds
        audio_s += prediction.runtime.audio_seconds
        print(f"{runner.name}: {index}/{len(rows)}", flush=True)
    status = "completed" if not failures else ("partial_failure" if hypotheses else "failed")
    wer = corpus_word_error_rate(references, hypotheses) if references else None
    cer = corpus_character_error_rate(references, hypotheses) if references else None
    crema_wer = CREMA_D_WER[runner.name]
    return {
        "runner": runner.name,
        "status": status,
        "licence_scope": NC_LABEL,
        "evaluated_clips": len(references),
        "failed_clips": len(failures),
        "metrics": {
            "wer": wer,
            "cer": cer,
            "crema_d_acted_us_english_wer": crema_wer,
            "wer_absolute_difference_vs_crema_d": wer - crema_wer if wer is not None else None,
            "normalization": "lowercase ASCII alphanumeric tokens; punctuation removed",
        },
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
        default=Path("data/manifests/ghana-english-wer.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/ghana-english-wer"),
    )
    parser.add_argument("--whisper-path", type=Path, required=True)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/ghana-english-wer-results.json"),
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
    results = [
        evaluate_runner(runner, rows, arguments.cache_dir)
        for runner in runners
    ]
    payload = {
        "report_version": "1",
        "title": "Ghanaian-English ASR slice — NC research-only",
        "licence_scope": NC_LABEL,
        "commercial_redistribution_prohibited": True,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": {
            "dataset": "ghananlpcommunity/ghana-english-asr-2700hrs",
            "dataset_revision": rows[0]["dataset_revision"],
            "licence": LICENCE,
            "attribution": rows[0]["attribution"],
            "manifest": str(arguments.manifest),
            "clip_count": len(rows),
            "duration_seconds": sum(float(row["duration_s"]) for row in rows),
            "sample_rate_hz": 16_000,
            "channels": 1,
            "speaker_disjoint": False,
            "speaker_limitation": (
                "The published corpus exposes no speaker IDs. Speaker-disjoint sampling "
                "and speaker leakage checks were therefore impossible."
            ),
            "fine_tuning_performed": False,
            "gate_decision": "not_evaluated_gate_remains_closed",
        },
        "execution": {
            "offline_after_fetch": True,
            "network_disabled_by_model_runtime_flags": True,
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
            "dataset": rows[0]["attribution"] + " CC BY-NC 4.0.",
            "whisper-small": MODEL_METADATA["whisper-small"]["attribution"],
            "sensevoice-small": MODEL_METADATA["sensevoice-small"]["attribution"],
        },
        "limitations": [
            "NC research-only: these results and manifest are not authorised for a "
            "commercially redistributed training set.",
            "The slice is deterministic but not speaker-disjoint because speaker IDs are absent.",
            "Broadcast-news source transcripts may contain proper-noun errors.",
            "CREMA-D is acted US English; its WER is context, not a matched-domain control.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {NC_LABEL} results to {arguments.output}")


if __name__ == "__main__":
    main()
