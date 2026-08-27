from __future__ import annotations

import importlib.util
import wave
from array import array
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "prepare_starss23_localization.py"
    spec = importlib.util.spec_from_file_location("prepare_starss23", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_partition_selection_balances_targets_and_limits_each_recording() -> None:
    script = load_script()
    rows = []
    for index in range(20):
        rows.append(
            {
                "events": [{"label": "laugh"}] if index < 12 else [],
                "laughter_overlap_frames": index,
                "overlap_frames": index,
                "max_polyphony": 2,
                "source_recording": f"mix{index // 2:02d}.wav",
                "window_start_ms": (index % 2) * 10_000,
                "official_partition": "dev-train-tau",
            }
        )

    selected = script.select_partition(rows, 10)

    assert len(selected) == 10
    assert sum(bool(row["events"]) for row in selected) == 6
    assert (
        max(
            sum(row["source_recording"] == source for row in selected)
            for source in {row["source_recording"] for row in selected}
        )
        <= 2
    )


def test_downmix_window_writes_16khz_mono_pcm(tmp_path: Path) -> None:
    script = load_script()
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    frames = array("h", [1000, -1000, 1000, -1000] * 240_000)
    with wave.open(str(source), "wb") as output:
        output.setparams((4, 2, 24_000, 0, "NONE", "not compressed"))
        output.writeframes(frames.tobytes())

    script.downmix_window(source, target, 0)

    with wave.open(str(target), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 16_000
        assert audio.getnframes() == 160_000
