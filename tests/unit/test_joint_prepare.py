from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from attune.training.prepare import load_source_rows, prepare_joint_features


class Tokenizer:
    def encode(self, text: str):
        return [len(text)]


class Frontend:
    tokenizer = Tokenizer()

    def __call__(self, wav: bytes):
        assert wav == b"audio"
        return np.zeros((8, 80), dtype=np.float32)


def test_prepare_joint_features_hashes_audio_and_outputs(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "clip_id": "one",
                "dataset_id": "fixture",
                "split": "train",
                "speaker_id": "speaker-one",
                "audio_path": "clip.wav",
                "audio_sha256": digest,
                "duration_ms": 700,
                "transcript": "hello",
            }
        )
        + "\n"
    )
    manifest = tmp_path / "manifests" / "joint.jsonl"
    manifest.parent.mkdir()
    rows = prepare_joint_features(
        load_source_rows(source),
        frontend=Frontend(),
        output_dir=tmp_path / "derived",
        manifest_path=manifest.resolve(),
    )

    assert rows[0].token_ids == [5]
    assert manifest.is_file()
    assert rows[0].feature_path.as_posix().startswith("../derived/features/")


def test_source_manifest_rejects_speaker_leakage(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    rows = []
    for split in ("train", "development"):
        rows.append(
            {
                "clip_id": split,
                "dataset_id": "fixture",
                "split": split,
                "speaker_id": "same-speaker",
                "audio_path": "clip.wav",
                "audio_sha256": digest,
                "duration_ms": 700,
            }
        )
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    try:
        load_source_rows(source)
    except ValueError as error:
        assert "crosses" in str(error)
    else:
        raise AssertionError("speaker leakage was not rejected")
