from __future__ import annotations

import csv
import importlib.util
import wave
from array import array
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "prepare_starss23_scene_raster.py"
    spec = importlib.util.spec_from_file_location("prepare_starss23_scene_raster", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def test_laughter_frames_union_sources_into_contiguous_attune_events() -> None:
    script = load_script()
    rows = [
        (10, 4, 1),
        (11, 4, 1),
        (11, 4, 2),
        (11, 0, 2),
        (13, 4, 1),
        (20, 4, 2),
    ]

    events = script.laughter_events(rows, window_start_ms=1000)

    assert events == [
        {
            "label": "laugh",
            "start_ms": 0,
            "end_ms": 200,
            "source_class": 4,
            "source_indices": [1, 2],
        },
        {
            "label": "laugh",
            "start_ms": 300,
            "end_ms": 400,
            "source_class": 4,
            "source_indices": [1],
        },
        {
            "label": "laugh",
            "start_ms": 1000,
            "end_ms": 1100,
            "source_class": 4,
            "source_indices": [2],
        },
    ]


def test_candidates_take_only_first_60s_and_drop_short_or_music_excerpts(tmp_path: Path) -> None:
    script = load_script()
    root = tmp_path / "metadata_dev"
    long_rows = [(frame, 0, 1) for frame in range(600)]
    long_rows[10] = (10, 4, 1)
    long_rows[12] = (12, 4, 1)
    write_csv(root / "dev-train-sony" / "fold3_room21_mix001.csv", long_rows)

    music_in_excerpt = [(frame, 0, 1) for frame in range(600)]
    music_in_excerpt[5] = (5, 8, 1)
    write_csv(root / "dev-train-sony" / "fold3_room22_mix001.csv", music_in_excerpt)

    music_after = [(frame, 0, 1) for frame in range(800)]
    music_after[700] = (700, 8, 1)
    write_csv(root / "dev-train-tau" / "fold4_room6_mix001.csv", music_after)

    short_rows = [(frame, 4, 1) for frame in range(400)]
    write_csv(root / "dev-test-tau" / "fold5_room2_mix001.csv", short_rows)

    test_rows = [(frame, 1, 1) for frame in range(600)]
    test_rows[20] = (20, 4, 2)
    write_csv(root / "dev-test-sony" / "fold6_room4_mix001.csv", test_rows)

    selected = script.candidates(root)
    names = {row["source_recording"] for row in selected}
    assert names == {
        "fold3_room21_mix001.wav",
        "fold4_room6_mix001.wav",
        "fold6_room4_mix001.wav",
    }
    assert all(row["window_start_ms"] == 0 for row in selected)
    by_name = {row["source_recording"]: row for row in selected}
    assert [event["start_ms"] for event in by_name["fold3_room21_mix001.wav"]["events"]] == [
        1000,
        1200,
    ]
    assert by_name["fold4_room6_mix001.wav"]["events"] == []
    development, inspection = script.official_partitions(selected)
    assert {row["source_recording"] for row in development} == {
        "fold3_room21_mix001.wav",
        "fold4_room6_mix001.wav",
    }
    assert {row["source_recording"] for row in inspection} == {"fold6_room4_mix001.wav"}


def test_downmix_window_writes_60s_16khz_mono_pcm(tmp_path: Path) -> None:
    script = load_script()
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    frames = array("h", [1000, -1000, 1000, -1000] * 1_440_000)
    with wave.open(str(source), "wb") as output:
        output.setparams((4, 2, 24_000, 0, "NONE", "not compressed"))
        output.writeframes(frames.tobytes())

    script.downmix_window(source, target, 0)

    with wave.open(str(target), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 16_000
        assert audio.getnframes() == 960_000
