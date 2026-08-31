from __future__ import annotations

import io
import wave

import numpy as np

from attune.inference.affect_fusion import (
    DEFAULT_AFFECT_CONFIDENCE_THRESHOLD,
    FusedAffectBackend,
    fuse_probabilities,
)
from attune.schema.migration import migrate_v1_to_v2
from attune.schema.output import AffectCategory, AttuneOutput


class FakeCadenceBackend:
    name = "cadence-fixture"

    def __init__(self, payload: dict) -> None:
        self.output = migrate_v1_to_v2(AttuneOutput.model_validate(payload))

    def analyse_wav(self, audio: bytes):
        return self.output


class FakeAffectStudent:
    labels = tuple(category.value for category in AffectCategory)
    parameter_count = 57_937_364
    quantization = "fp32"

    def predict_probabilities(self, payloads):
        assert all(payload[:4] == b"RIFF" for payload in payloads)
        prediction = np.array([0.05, 0.75, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])
        return np.stack([prediction] * len(payloads))


class MixedAffectStudent(FakeAffectStudent):
    def predict_probabilities(self, payloads):
        if len(payloads) == 1:
            return np.array([[0.05, 0.05, 0.05, 0.05, 0.72, 0.04, 0.03, 0.01]])
        return np.array(
            [
                [0.05, 0.72, 0.05, 0.05, 0.05, 0.04, 0.03, 0.01],
                [0.05, 0.05, 0.72, 0.05, 0.05, 0.04, 0.03, 0.01],
            ]
        )


def _wav(duration_ms: int = 1800) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * round(duration_ms * 16))
    return target.getvalue()


def test_probability_fusion_preserves_normalization() -> None:
    cadence = np.array([0.8, 0.2])
    student = np.array([0.1, 0.9])

    fused = fuse_probabilities(cadence, student, acoustic_weight=0.75)

    assert np.allclose(fused, [0.275, 0.725])
    assert np.isclose(fused.sum(), 1.0)


def test_backend_uses_selective_default_threshold(example_payload: dict) -> None:
    backend = FusedAffectBackend(
        FakeCadenceBackend(example_payload),
        FakeAffectStudent(),
        acoustic_weight=0.0,
    )

    assert backend.confidence_threshold == DEFAULT_AFFECT_CONFIDENCE_THRESHOLD


def test_backend_replaces_affect_but_preserves_transcript(example_payload: dict) -> None:
    backend = FusedAffectBackend(
        FakeCadenceBackend(example_payload),
        FakeAffectStudent(),
        acoustic_weight=0.97,
        confidence_threshold=0.25,
    )

    output = backend.analyse_wav(_wav())

    assert output.transcript.text == "I said leave me alone"
    assert output.affect.top_label == AffectCategory.JOY
    assert output.affect.abstain is False
    assert output.affect_spans[0].top_label == AffectCategory.JOY
    assert output.model.name == "attune-cadence-300m-affect-fusion"
    assert output.model.quantization.endswith("+fp32-affect")
    assert output.uncertainty.out_of_distribution_available is False
    assert output.uncertainty.out_of_distribution_probability is None


def test_backend_abstains_when_confident_spans_disagree(example_payload: dict) -> None:
    cadence = FakeCadenceBackend(example_payload)
    affect = cadence.output.affect
    cadence.output = cadence.output.model_copy(
        update={
            "affect_spans": [
                affect.model_copy(update={"start_ms": 0, "end_ms": 900}),
                affect.model_copy(update={"start_ms": 900, "end_ms": 1800}),
            ]
        }
    )
    backend = FusedAffectBackend(
        cadence,
        MixedAffectStudent(),
        acoustic_weight=1.0,
        confidence_threshold=0.25,
    )

    output = backend.analyse_wav(_wav())

    assert [span.top_label for span in output.affect_spans] == [
        AffectCategory.JOY,
        AffectCategory.DISTRESS,
    ]
    assert output.affect.abstain is True
    assert output.affect.top_label is None
    assert output.affect.abstention_reason == "mixed_affect_spans"
