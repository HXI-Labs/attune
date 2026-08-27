from __future__ import annotations

import importlib.util
import json
import math
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/train_probe.py"
SPEC = importlib.util.spec_from_file_location("train_probe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN_PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_PROBE)

SOURCE_LABELS = {
    "laughter": "laugh",
    "sigh": "sigh",
    "cough": "cough",
    "throatclearing": "throat_clear",
    "sneeze": "sneeze",
}


def write_tone(path: Path, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = array(
        "h",
        (
            round(8_000 * math.sin(2 * math.pi * frequency * index / 16_000))
            for index in range(1_600)
        ),
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


@pytest.mark.integration
def test_probe_trains_only_a_linear_head_on_synthetic_wavs(tmp_path: Path) -> None:
    dataset = tmp_path / "vocalsound"
    for speaker_index in range(6):
        speaker = f"f{speaker_index:04d}"
        for label_index, source_label in enumerate(SOURCE_LABELS):
            write_tone(
                dataset / f"{speaker}_0_{source_label}.wav",
                200 + label_index * 100 + speaker_index,
            )

    inspection_cache = tmp_path / "inspection"
    manifest = tmp_path / "inspection.jsonl"
    rows = []
    for speaker_index, speaker in enumerate(("f9001", "m9002")):
        for label_index, (source_label, attune_label) in enumerate(SOURCE_LABELS.items()):
            relative_path = f"vocalsound/{speaker}_0_{source_label}.wav"
            write_tone(
                inspection_cache / relative_path,
                200 + label_index * 100 + speaker_index,
            )
            rows.append(
                {
                    "source_dataset": "VocalSound",
                    "source_filename": Path(relative_path).name,
                    "speaker_id": f"vocalsound:{speaker}",
                    "split": "inspect" if speaker_index == 0 else "held_out_speakers",
                    "cache_path": relative_path,
                    "intended_attune_labels": {"events": [attune_label]},
                }
            )
    manifest.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    fsd50k_cache = tmp_path / "fsd50k"
    fsd50k_manifest = tmp_path / "fsd50k.jsonl"
    fsd50k_rows = []
    for label_index, source_class in enumerate(
        ("Shout", "Whispering", "Crying_and_sobbing", "Screaming")
    ):
        relative_path = f"fsd50k/ood-{label_index}.wav"
        write_tone(fsd50k_cache / relative_path, 900 + label_index * 100)
        fsd50k_rows.append(
            {
                "cache_path": relative_path,
                "clip_id": f"ood-{label_index}",
                "partition": "validation",
                "source_class": source_class,
            }
        )
    fsd50k_manifest.write_text(
        "".join(f"{json.dumps(row)}\n" for row in fsd50k_rows),
        encoding="utf-8",
    )

    report = TRAIN_PROBE.train(
        SimpleNamespace(
            dataset_dir=dataset,
            inspection_manifest=manifest,
            inspection_cache=inspection_cache,
            fsd50k_probe_manifest=fsd50k_manifest,
            fsd50k_probe_cache=fsd50k_cache,
            test_set="inspection",
            validation_fraction=0.33,
            min_train_clips=10,
            epochs=2,
            patience=2,
            batch_size=8,
            learning_rate=1e-2,
            seed=0,
            checkpoint_output=tmp_path / "head.pt",
        )
    )

    assert report["encoder_frozen"] is True
    assert report["embedding"]["trainable_parameters"] == 0
    assert report["head"]["type"] == "linear"
    assert report["partitions"]["train"]["clips"] == 20
    assert report["partitions"]["validation"]["clips"] == 10
    assert report["partitions"]["test"]["clips"] == 10
    assert set(report["partitions"]["train"]["speakers"]).isdisjoint(
        report["partitions"]["test"]["speakers"]
    )
    assert (tmp_path / "head.pt").is_file()
