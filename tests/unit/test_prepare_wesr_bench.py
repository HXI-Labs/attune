import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from attune.training.prepare import load_source_rows

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "prepare_wesr_bench.py"
SPEC = importlib.util.spec_from_file_location("prepare_wesr_bench", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_wesr_markup_is_removed_without_joining_words() -> None:
    sentence = "Please <whispering>do not [cough] call</whispering> again."

    assert MODULE._transcript(sentence) == "Please do not call again."


def test_wesr_labels_keep_first_occurrence_order() -> None:
    assert MODULE._ordered_unique(["laugh", "cough", "laugh"]) == ["laugh", "cough"]


def test_wesr_continuous_events_map_to_speech_styles() -> None:
    assert MODULE.CONTINUOUS_STYLE_MAP["crying"] == "crying_speech"
    assert MODULE.CONTINUOUS_STYLE_MAP["laughing"] == "laughing_speech"


def test_wesr_runtime_inventory_is_an_explicit_ontology_subset() -> None:
    assert "breath" in MODULE.DISCRETE_EVENT_MAP.values()
    assert "breath" not in MODULE.SUPPORTED_EVENT_VALUES
    assert "crying_speech" in MODULE.CONTINUOUS_STYLE_MAP.values()
    assert "crying_speech" not in MODULE.SUPPORTED_STYLE_VALUES


def test_wesr_manifest_paths_resolve_from_the_manifest(tmp_path: Path) -> None:
    audio_bytes = b"test-flac"
    parquet_path = tmp_path / "download" / "english.parquet"
    parquet_path.parent.mkdir()
    pq.write_table(
        pa.table(
            {
                "audio": [{"bytes": audio_bytes, "path": "clip.flac"}],
                "sentence": ["Please <whispering>wait</whispering> [cough]"],
                "duration": [1.25],
            }
        ),
        parquet_path,
    )
    source_manifest = tmp_path / "manifests" / "wesr.source.jsonl"
    annotations = tmp_path / "manifests" / "wesr.annotations.jsonl"
    audio_dir = tmp_path / "raw" / "audio"

    MODULE.prepare(parquet_path, audio_dir, source_manifest, annotations)

    source_row = json.loads(source_manifest.read_text())
    assert source_row["audio_path"] == "../raw/audio/clip.flac"
    loaded = load_source_rows(source_manifest)
    assert loaded[0].audio_path == (audio_dir / "clip.flac").resolve()
    assert loaded[0].event_presence == ["cough"]
    assert loaded[0].styles == ["whispering"]
