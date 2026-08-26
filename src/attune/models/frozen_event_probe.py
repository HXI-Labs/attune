"""Speaker-disjoint VocalSound data utilities for the frozen Stage 2 event probe."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from attune.schema.output import EventLabel

EVENT_LABELS = (
    EventLabel.LAUGH,
    EventLabel.SIGH,
    EventLabel.COUGH,
    EventLabel.THROAT_CLEAR,
    EventLabel.SNEEZE,
)
VOCALSOUND_TO_EVENT = {
    "laughter": EventLabel.LAUGH,
    "sigh": EventLabel.SIGH,
    "cough": EventLabel.COUGH,
    "throatclearing": EventLabel.THROAT_CLEAR,
    "sneeze": EventLabel.SNEEZE,
}


class ProbeDataError(RuntimeError):
    """Raised when probe data would violate a required split invariant."""


@dataclass(frozen=True)
class ProbeExample:
    """One source-labelled VocalSound clip."""

    path: Path
    speaker_id: str
    label: EventLabel


@dataclass(frozen=True)
class ProbeSplit:
    """Speaker-disjoint train, validation, and inspection-test examples."""

    train: tuple[ProbeExample, ...]
    validation: tuple[ProbeExample, ...]
    test: tuple[ProbeExample, ...]

    def speaker_ids(self, partition: str) -> set[str]:
        return {example.speaker_id for example in getattr(self, partition)}


def vocalsound_label(filename: str | Path) -> EventLabel:
    """Map an official VocalSound filename suffix to the Attune event ontology."""
    source_label = Path(filename).stem.rsplit("_", 1)[-1].lower()
    try:
        return VOCALSOUND_TO_EVENT[source_label]
    except KeyError as error:
        raise ProbeDataError(f"unsupported VocalSound label in {filename!s}") from error


def vocalsound_speaker_id(filename: str | Path) -> str:
    """Return the namespaced speaker ID encoded by a VocalSound filename."""
    source_speaker = Path(filename).stem.split("_", 1)[0].lower()
    if not source_speaker or source_speaker[0] not in {"f", "m"}:
        raise ProbeDataError(f"cannot parse VocalSound speaker from {filename!s}")
    return f"vocalsound:{source_speaker}"


def load_inspection_rows(manifest: Path) -> list[dict[str, Any]]:
    """Load only VocalSound rows from the committed inspection manifest."""
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProbeDataError(f"cannot read inspection manifest {manifest}: {error}") from error

    rows = [
        row
        for line in lines
        if line.strip()
        and isinstance((row := json.loads(line)), dict)
        and row.get("source_dataset") == "VocalSound"
    ]
    if not rows:
        raise ProbeDataError(f"no VocalSound rows found in {manifest}")
    return rows


def inspection_examples(
    manifest: Path,
    cache_root: Path,
    *,
    split: str | None = None,
) -> list[ProbeExample]:
    """Build the immutable inspection test set from manifest cache paths."""
    rows = load_inspection_rows(manifest)
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]
    examples = [
        ProbeExample(
            path=cache_root / row["cache_path"],
            speaker_id=row["speaker_id"],
            label=EventLabel(row["intended_attune_labels"]["events"][0]),
        )
        for row in rows
    ]
    if not examples:
        raise ProbeDataError(f"no VocalSound inspection rows selected for split {split!r}")
    return examples


def discover_vocalsound(dataset_root: Path) -> list[ProbeExample]:
    """Discover the five supported 16 kHz VocalSound classes below a local root."""
    if not dataset_root.is_dir():
        raise ProbeDataError(f"VocalSound directory does not exist: {dataset_root}")

    examples: list[ProbeExample] = []
    for path in sorted(dataset_root.rglob("*.wav")):
        try:
            label = vocalsound_label(path.name)
            speaker_id = vocalsound_speaker_id(path.name)
        except ProbeDataError:
            continue
        examples.append(ProbeExample(path=path, speaker_id=speaker_id, label=label))
    if not examples:
        raise ProbeDataError(f"no supported VocalSound WAV files found below {dataset_root}")
    return examples


def make_speaker_disjoint_split(
    candidates: list[ProbeExample],
    test: list[ProbeExample],
    *,
    excluded_speakers: set[str],
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> ProbeSplit:
    """Exclude inspection speakers and partition remaining speakers into train/validation."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    eligible = [example for example in candidates if example.speaker_id not in excluded_speakers]
    speakers = sorted({example.speaker_id for example in eligible})
    if len(speakers) < 2:
        raise ProbeDataError("at least two non-inspection speakers are required")

    random.Random(seed).shuffle(speakers)
    validation_count = max(1, round(len(speakers) * validation_fraction))
    validation_count = min(validation_count, len(speakers) - 1)
    validation_speakers = set(speakers[:validation_count])
    train = tuple(example for example in eligible if example.speaker_id not in validation_speakers)
    validation = tuple(example for example in eligible if example.speaker_id in validation_speakers)
    result = ProbeSplit(train=train, validation=validation, test=tuple(test))
    assert_speaker_disjoint(result, excluded_speakers)
    return result


def assert_speaker_disjoint(split: ProbeSplit, excluded_speakers: set[str]) -> None:
    """Fail closed if any speaker crosses train, validation, or inspection test."""
    train = split.speaker_ids("train")
    validation = split.speaker_ids("validation")
    test = split.speaker_ids("test")
    if train & validation or train & test or validation & test:
        raise ProbeDataError("a speaker crosses train, validation, and/or test partitions")
    if (train | validation) & excluded_speakers:
        raise ProbeDataError("an inspection speaker appears in probe training or validation")
