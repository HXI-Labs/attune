#!/usr/bin/env python3
"""Regenerate deterministic synthetic PCM16 fixture audio."""

from __future__ import annotations

import math
import random
import wave
from array import array
from pathlib import Path

SAMPLE_RATE = 16_000
DURATION_SECONDS = 0.5
SPECS = {
    "neutral_words_joy_delivery.wav": (660, 0.35, 1),
    "neutral_words_anger_delivery.wav": (180, 0.75, 2),
    "explicit_match_joy.wav": (720, 0.4, 3),
    "explicit_conflict_anger.wav": (150, 0.8, 4),
    "same_text_joy.wav": (600, 0.3, 5),
    "same_text_distress.wav": (240, 0.2, 6),
}


def main() -> None:
    root = Path(__file__).parent
    frame_count = round(SAMPLE_RATE * DURATION_SECONDS)
    for filename, (frequency, amplitude, seed) in SPECS.items():
        randomizer = random.Random(seed)
        samples = array(
            "h",
            (
                round(
                    32767
                    * max(
                        -1,
                        min(
                            1,
                            amplitude * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)
                            + 0.015 * randomizer.uniform(-1, 1),
                        ),
                    )
                )
                for index in range(frame_count)
            ),
        )
        with wave.open(str(root / filename), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(samples.tobytes())


if __name__ == "__main__":
    main()
