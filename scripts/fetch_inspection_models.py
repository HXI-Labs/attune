#!/usr/bin/env python3
"""Fetch the two reviewed baseline checkpoints at pinned revisions with finite timeouts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MODELS = {
    "whisper-small": {
        "repository": "openai/whisper-small",
        "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
        "attribution": "OpenAI Whisper-Small; upstream Whisper MIT licence.",
    },
    "sensevoice-small": {
        "repository": "FunAudioLLM/SenseVoiceSmall",
        "revision": "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
        "attribution": (
            "SenseVoiceSmall by FunASR/FunAudioLLM; "
            "FunASR Model Open Source License Agreement v1.1."
        ),
    },
    "emotion2vec-plus": {
        "repository": "emotion2vec/emotion2vec_plus_base",
        "revision": "b318240bfe67db81a8c572ecb37ce9c3759b81c9",
        "attribution": (
            "emotion2vec+ base by emotion2vec and FunASR/FunAudioLLM; "
            "FunASR Model Open Source License Agreement."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/model-cache"))
    parser.add_argument("--model", action="append", choices=tuple(MODELS))
    parser.add_argument("--etag-timeout", type=int, default=20)
    parser.add_argument("--download-timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "error: huggingface_hub is required; install the model-runners environment"
        ) from error

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(arguments.etag_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(arguments.download_timeout)
    token = os.environ.get("HF_TOKEN")
    failures: list[str] = []
    for name in arguments.model or list(MODELS):
        metadata = MODELS[name]
        target = arguments.output_dir / name
        print(
            f"Fetching {name} revision {metadata['revision']} — {metadata['attribution']}",
            flush=True,
        )
        try:
            snapshot_download(
                repo_id=metadata["repository"],
                revision=metadata["revision"],
                local_dir=target,
                token=token,
                etag_timeout=arguments.etag_timeout,
                max_workers=2,
            )
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
    if failures:
        raise SystemExit("model fetch incomplete:\n- " + "\n- ".join(failures))
    print(f"Fetched reviewed checkpoints under {arguments.output_dir}")


if __name__ == "__main__":
    main()
