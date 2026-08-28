from __future__ import annotations

import csv
import importlib.util
import inspect
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


def write_wav(path: Path, duration_ms: int, rate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = duration_ms * rate // 1000
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, nframes, "NONE", "not compressed"))
        output.writeframes(b"\x00\x00" * nframes)


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


def test_window_starts_require_wav_and_csv_coverage() -> None:
    script = load_script()
    assert script.window_starts_ms(60_000, 91_900) == [0]
    assert script.window_starts_ms(180_000, 90_000) == [0]
    assert script.window_starts_ms(180_000, 180_000) == [0, 60_000, 120_000]
    assert script.window_starts_ms(119_000, 200_000) == [0]
    assert script.window_starts_ms(59_900, 120_000) == []
    assert script.window_starts_ms(120_000, 59_900) == []


def test_candidates_skip_unlabeled_wav_tail_and_require_wav_csv(tmp_path: Path) -> None:
    script = load_script()
    metadata = tmp_path / "metadata_dev"
    audio = tmp_path / "mic_dev"
    long_rows = [(frame, 0, 1) for frame in range(600)]
    long_rows[10] = (10, 4, 1)
    write_csv(metadata / "dev-train-sony" / "fold3_room21_mix001.csv", long_rows)
    write_wav(audio / "dev-train-sony" / "fold3_room21_mix001.wav", 91_900)

    extra = [(frame, 0, 1) for frame in range(1800)]
    extra[10] = (10, 4, 1)
    extra[700] = (700, 4, 1)
    write_csv(metadata / "dev-train-sony" / "fold3_room22_mix001.csv", extra)
    write_wav(audio / "dev-train-sony" / "fold3_room22_mix001.wav", 90_000)

    test_rows = [(frame, 4, 1) for frame in range(600)]
    write_csv(metadata / "dev-test-sony" / "fold6_room4_mix001.csv", test_rows)
    write_wav(audio / "dev-test-sony" / "fold6_room4_mix001.wav", 60_000)

    selected = script.candidates(metadata, audio)
    by_key = {(row["source_recording"], row["window_start_ms"]): row for row in selected}
    assert set(by_key) == {
        ("fold3_room21_mix001.wav", 0),
        ("fold3_room22_mix001.wav", 0),
        ("fold6_room4_mix001.wav", 0),
    }
    assert ("fold3_room21_mix001.wav", 60_000) not in by_key
    assert ("fold3_room22_mix001.wav", 60_000) not in by_key
    assert ("fold3_room22_mix001.wav", 120_000) not in by_key


def test_candidates_tile_non_overlapping_60s_and_drop_music_windows(tmp_path: Path) -> None:
    script = load_script()
    root = tmp_path / "metadata_dev"
    long_rows = [(frame, 0, 1) for frame in range(1800)]
    long_rows[10] = (10, 4, 1)
    long_rows[12] = (12, 4, 1)
    long_rows[610] = (610, 4, 1)
    long_rows[1210] = (1210, 4, 2)
    write_csv(root / "dev-train-sony" / "fold3_room21_mix001.csv", long_rows)

    music_in_second = [(frame, 0, 1) for frame in range(1800)]
    music_in_second[650] = (650, 8, 1)
    music_in_second[10] = (10, 4, 1)
    write_csv(root / "dev-train-sony" / "fold3_room22_mix001.csv", music_in_second)

    short_rows = [(frame, 4, 1) for frame in range(400)]
    write_csv(root / "dev-test-tau" / "fold5_room2_mix001.csv", short_rows)

    test_rows = [(frame, 1, 1) for frame in range(600)]
    test_rows[20] = (20, 4, 2)
    write_csv(root / "dev-test-sony" / "fold6_room4_mix001.csv", test_rows)

    selected = script.candidates(root)
    assert all(row["window_start_ms"] % script.SCENE_MS == 0 for row in selected)
    assert all(
        event["start_ms"] >= 0 and event["end_ms"] <= script.SCENE_MS
        for row in selected
        for event in row["events"]
    )
    by_key = {(row["source_recording"], row["window_start_ms"]): row for row in selected}
    assert set(by_key) == {
        ("fold3_room21_mix001.wav", 0),
        ("fold3_room21_mix001.wav", 60_000),
        ("fold3_room21_mix001.wav", 120_000),
        ("fold3_room22_mix001.wav", 0),
        ("fold3_room22_mix001.wav", 120_000),
        ("fold6_room4_mix001.wav", 0),
    }
    assert ("fold3_room22_mix001.wav", 60_000) not in by_key
    first_window = by_key[("fold3_room21_mix001.wav", 0)]
    second_window = by_key[("fold3_room21_mix001.wav", 60_000)]
    assert [event["start_ms"] for event in first_window["events"]] == [1000, 1200]
    assert [event["start_ms"] for event in second_window["events"]] == [1000]
    development, inspection = script.official_partitions(selected)
    assert {row["source_recording"] for row in development} == {
        "fold3_room21_mix001.wav",
        "fold3_room22_mix001.wav",
    }
    assert {row["source_recording"] for row in inspection} == {"fold6_room4_mix001.wav"}
    assert {row["room"] for row in development}.isdisjoint({row["room"] for row in inspection})


def test_clipped_spanning_events_dropped_on_later_tiles_kept_on_first_60s() -> None:
    script = load_script()
    frames = [(frame, 4, 1) for frame in range(590, 646)]
    frames.extend((frame, 4, 1) for frame in range(700, 710))
    first, first_dropped = script.scored_laughter_events(frames, window_start_ms=0)
    later, later_dropped = script.scored_laughter_events(frames, window_start_ms=60_000)
    assert first_dropped == 0
    assert first == [
        {
            "label": "laugh",
            "start_ms": 59_000,
            "end_ms": 60_000,
            "source_class": 4,
            "source_indices": [1],
        }
    ]
    assert later_dropped == 1
    assert later == [
        {
            "label": "laugh",
            "start_ms": 10_000,
            "end_ms": 11_000,
            "source_class": 4,
            "source_indices": [1],
        }
    ]
    truncated = script.laughter_events(frames, window_start_ms=0)
    assert truncated[0]["end_ms"] == 60_000


def test_windows_do_not_cross_the_60s_boundary() -> None:
    script = load_script()
    starts = script.window_starts_ms(185_000, 185_000)
    assert starts == [0, 60_000, 120_000]
    for start in starts:
        end = start + script.SCENE_MS
        assert end % script.SCENE_MS == 0
        assert start % script.SCENE_MS == 0
        assert end - start == script.SCENE_MS
    assert script.window_starts_ms(59_900, 185_000) == []
    frames = [(frame, 4, 1) for frame in (599, 600)]
    first, first_dropped = script.scored_laughter_events(frames, window_start_ms=0)
    second, second_dropped = script.scored_laughter_events(frames, window_start_ms=60_000)
    assert first_dropped == 0
    assert first == [
        {
            "label": "laugh",
            "start_ms": 59_900,
            "end_ms": 60_000,
            "source_class": 4,
            "source_indices": [1],
        }
    ]
    assert second_dropped == 1
    assert second == []


def test_clip_ids_include_window_start_ms_and_do_not_overwrite() -> None:
    script = load_script()
    first = script.clip_id_for("development", "fold3_room21_mix001.wav", 0)
    second = script.clip_id_for("development", "fold3_room21_mix001.wav", 60_000)
    other = script.clip_id_for("development", "fold3_room22_mix001.wav", 0)
    assert "w000000" in first
    assert "w060000" in second
    assert first != second
    assert first != other
    sequential = "starss23-scene-development-001"
    assert first != sequential
    assert "{index:03d}" not in inspect.getsource(script.materialize)
    assert "clip_id_for" in inspect.getsource(script.materialize)


def test_mean_of_4_downmix_not_max_rms(tmp_path: Path) -> None:
    script = load_script()
    assert not hasattr(script, "choose_max_rms_channel")
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    # Constant [8000, 0, 0, 0]; mean-of-4 is 2000, max-RMS would keep 8000.
    frames = array("h", [8000, 0, 0, 0] * 1_440_000)
    with wave.open(str(source), "wb") as output:
        output.setparams((4, 2, 24_000, 0, "NONE", "not compressed"))
        output.writeframes(frames.tobytes())

    assert script.downmix_window(source, target, 0) is None

    with wave.open(str(target), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 16_000
        assert audio.getnframes() == 960_000
        samples = array("h", audio.readframes(audio.getnframes()))
    midpoint = samples[len(samples) // 2]
    assert 1500 <= midpoint <= 2500
    assert abs(midpoint - 8000) > 4000
    downmix_source = inspect.getsource(script.downmix_window)
    assert "choose_max_rms_channel" not in downmix_source
    assert "samples[chosen" not in downmix_source
    assert "/ SOURCE_CHANNELS" in downmix_source or "/ 4" in downmix_source
