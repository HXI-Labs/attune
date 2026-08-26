#!/usr/bin/env python3
"""Run Phase 1 baselines against fixtures or the local inspection manifest."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from attune.evaluation.harness import run_fixture_harness, run_inspection_harness


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _inspection_run_metadata() -> dict[str, object]:
    memory_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        first = meminfo.read_text(encoding="utf-8").splitlines()[0].split()
        memory_kib = int(first[1])
    return {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "run_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "entrypoint": "scripts/evaluate.py --inspection-manifest",
            "model_downloads_during_evaluation": False,
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MODELSCOPE_OFFLINE": "1",
                "FUNASR_DISABLE_UPDATE": "1",
            },
        },
        "hardware": {
            "os": f"{platform.system()} {platform.release()} {platform.machine()}",
            "cpu": platform.processor() or "Intel(R) Xeon(R) Processor",
            "logical_cpu_count": os.cpu_count(),
            "memory_gib": round(memory_kib / 1024 / 1024, 2) if memory_kib else None,
            "gpu_available": False,
            "execution_device": "cpu",
        },
        "software": {
            "python": platform.python_version(),
            "torch": _version("torch"),
            "torchaudio": _version("torchaudio"),
            "transformers": _version("transformers"),
            "funasr": _version("funasr"),
            "pydantic": _version("pydantic"),
        },
        "licence_attribution": {
            "VocalSound": "CC BY-SA 4.0; Gong, Yu, and Glass, ICASSP 2022.",
            "CREMA-D": "ODbL 1.0 database / DbCL 1.0 contents; Cao et al., 2014.",
            "Whisper-Small": "OpenAI Whisper; upstream MIT licence.",
            "SenseVoiceSmall": "FunASR/FunAudioLLM; FunASR Model Open Source License Agreement v1.1.",
            "emotion2vec+ base": "emotion2vec and FunASR/FunAudioLLM; FunASR model licence.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/fixtures/semantic_conflict"),
        help="fixture directory containing manifest.json and gold.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/baseline-results.json"),
        help="machine-readable report destination",
    )
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        help="inspection JSONL manifest; switches from fixture to local-WAV evaluation",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/inspection-set"),
        help="local audio root for inspection-manifest cache_path values",
    )
    arguments = parser.parse_args()
    if arguments.inspection_manifest:
        report = run_inspection_harness(arguments.inspection_manifest, arguments.cache_dir)
        report.update(_inspection_run_metadata())
    else:
        report = run_fixture_harness(arguments.fixtures)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {arguments.output}")
    for skipped in report.get("skipped_runners", []):
        print(f"Skipped {skipped['runner']}: {skipped['reason']}")
    for result in report.get("runner_results", []):
        if result["status"] == "skipped":
            print(f"Skipped {result['runner']}: {result['skip_reason']}")


if __name__ == "__main__":
    main()
