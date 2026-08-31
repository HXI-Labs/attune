#!/usr/bin/env python3
"""Extract the English WESR-Bench parquet into Attune evaluation source rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq

from attune.models.joint import SUPPORTED_EVENTS, SUPPORTED_STYLES
from attune.training.prepare import SourceRow

DISCRETE_EVENT_MAP = {
    "cough": "cough",
    "laughs": "laugh",
    "laughing": "laugh",
    "crowd_laughter": "laugh",
    "chuckle": "laugh",
    "giggle": "laugh",
    "chortle": "laugh",
    "sobbing": "sob",
    "cry": "sob",
    "sigh": "sigh",
    "clear_throat": "throat_clear",
    "scream": "scream",
    "inhale": "breath",
    "breathing": "breath",
    "exhale": "breath",
}
CONTINUOUS_STYLE_MAP = {
    "shouting": "shouting",
    "whispering": "whispering",
    "laughing": "laughing_speech",
    "crying": "crying_speech",
    "singing": "singing",
}
SUPPORTED_EVENT_VALUES = {label.value for label in SUPPORTED_EVENTS}
SUPPORTED_STYLE_VALUES = {label.value for label in SUPPORTED_STYLES}
DISCRETE_TAG = re.compile(r"\[([^]]+)]")
OPENING_TAG = re.compile(r"<([^/>]+)>")
MARKUP = re.compile(r"\[[^]]+]|</?[^>]+>")


def _labels(pattern: re.Pattern[str], sentence: str) -> list[str]:
    return [match.strip().lower() for match in pattern.findall(sentence)]


def _ordered_unique(labels: list[str]) -> list[str]:
    return list(dict.fromkeys(labels))


def _transcript(sentence: str) -> str:
    return " ".join(MARKUP.sub(" ", sentence).split())


def prepare(
    parquet_path: Path,
    audio_dir: Path,
    source_manifest: Path,
    annotations_path: Path,
) -> None:
    table = pq.read_table(parquet_path)
    audio_dir.mkdir(parents=True, exist_ok=True)
    source_rows = []
    annotations = []
    for index, row in enumerate(table.to_pylist(), start=1):
        sentence = row["sentence"]
        discrete = _labels(DISCRETE_TAG, sentence)
        continuous = _labels(OPENING_TAG, sentence)
        ontology_events = _ordered_unique(
            [DISCRETE_EVENT_MAP[label] for label in discrete if label in DISCRETE_EVENT_MAP]
        )
        ontology_styles = _ordered_unique(
            [CONTINUOUS_STYLE_MAP[label] for label in continuous if label in CONTINUOUS_STYLE_MAP]
        )
        events = [label for label in ontology_events if label in SUPPORTED_EVENT_VALUES]
        styles = [label for label in ontology_styles if label in SUPPORTED_STYLE_VALUES]
        audio = row["audio"]
        filename = Path(audio["path"]).name
        audio_path = audio_dir / filename
        audio_bytes = audio["bytes"]
        if audio_path.is_file() and audio_path.read_bytes() != audio_bytes:
            raise ValueError(f"existing WESR audio differs: {audio_path}")
        audio_path.write_bytes(audio_bytes)
        negative_tasks = []
        if not events:
            negative_tasks.append("event_presence")
        if not styles:
            negative_tasks.append("styles")
        clip_id = f"wesr-en-{index:04d}"
        source_rows.append(
            SourceRow(
                clip_id=clip_id,
                dataset_id="wesr_bench_en_v0.1",
                split="sealed_test",
                speaker_id=None,
                audio_path=Path(
                    os.path.relpath(audio_path.resolve(), source_manifest.parent.resolve())
                ),
                audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
                duration_ms=round(float(row["duration"]) * 1000),
                transcript=_transcript(sentence),
                event_presence=events,
                styles=styles,
                is_ood=True,
                auxiliary_negative_tasks=negative_tasks,
            )
        )
        annotations.append(
            {
                "clip_id": clip_id,
                "sentence_markup": sentence,
                "discrete_tags": discrete,
                "continuous_tags": continuous,
                "attune_ontology_events": ontology_events,
                "attune_ontology_styles": ontology_styles,
                "attune_event_presence": events,
                "attune_styles": styles,
            }
        )
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text("".join(row.model_dump_json() + "\n" for row in source_rows))
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(
        arguments.parquet,
        arguments.audio_dir,
        arguments.source_manifest,
        arguments.annotations,
    )


if __name__ == "__main__":
    main()
