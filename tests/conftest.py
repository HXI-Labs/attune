from __future__ import annotations

import pytest

from attune.schema.output import AttuneOutput


@pytest.fixture
def example_payload() -> dict:
    return {
        "schema_version": "1.0",
        "model": {"name": "phase-0-placeholder", "version": "0", "quantization": None},
        "audio": {
            "duration_ms": 1800,
            "sample_rate_hz": 16000,
            "channels": 1,
            "quality": {"clipping": 0.01, "low_snr": 0.08, "far_field": 0.03},
        },
        "language": {"label": "en", "confidence": 0.98},
        "transcript": {
            "text": "I said leave me alone",
            "confidence": 0.93,
            "words": [
                {"id": "w1", "text": "I", "start_ms": 0, "end_ms": 100, "confidence": 0.99},
                {
                    "id": "w2",
                    "text": "said",
                    "start_ms": 140,
                    "end_ms": 350,
                    "confidence": 0.98,
                },
                {
                    "id": "w3",
                    "text": "leave",
                    "start_ms": 410,
                    "end_ms": 690,
                    "confidence": 0.96,
                },
                {
                    "id": "w4",
                    "text": "me",
                    "start_ms": 720,
                    "end_ms": 850,
                    "confidence": 0.97,
                },
                {
                    "id": "w5",
                    "text": "alone",
                    "start_ms": 880,
                    "end_ms": 1250,
                    "confidence": 0.95,
                },
            ],
        },
        "styles": [
            {
                "id": "s1",
                "label": "shouting",
                "start_ms": 410,
                "end_ms": 1250,
                "start_word_id": "w3",
                "end_word_id": "w5",
                "confidence": 0.88,
                "status": "committed",
            }
        ],
        "events": [
            {
                "id": "e1",
                "label": "sob",
                "start_ms": 1360,
                "end_ms": 1540,
                "after_word_id": "w5",
                "confidence": 0.84,
                "status": "committed",
            }
        ],
        "affect": {
            "start_ms": 0,
            "end_ms": 1540,
            "valence": {"value": -0.65, "confidence": 0.72},
            "arousal": {"value": 0.79, "confidence": 0.81},
            "dominance": {"value": 0.18, "confidence": 0.58},
            "categories": {
                "neutral": 0.02,
                "joy": 0.01,
                "distress": 0.55,
                "anger": 0.28,
                "fear": 0.05,
                "surprise": 0.01,
                "other": 0.03,
                "ambiguous": 0.05,
            },
            "top_label": "distress",
            "top_label_confidence": 0.55,
            "abstain": False,
        },
        "uncertainty": {"out_of_distribution_probability": 0.12},
    }


@pytest.fixture
def example_output(example_payload: dict) -> AttuneOutput:
    return AttuneOutput.model_validate(example_payload)
