from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from attune.inference.export import OUTPUT_NAMES, PROBE_EMBEDDING_OUTPUT
from attune.inference.onnx_backend import (
    GreedySenseVoiceDecoder,
    OnnxAttuneBackend,
    RuntimeCalibration,
)
from attune.inference.probe_head import ProbeDecision


class FakeSession:
    def run(self, names, inputs):
        assert tuple(names) == OUTPUT_NAMES
        assert inputs["speech"].shape[0] == 1
        assert inputs["speech"].shape[2] == 80
        event = np.full((1, 10, 7), -8.0, dtype=np.float32)
        event[0, 2:6, 0] = 8.0
        style = np.asarray([[8.0, -8.0]], dtype=np.float32)
        affect = np.full((1, 8), -3.0, dtype=np.float32)
        affect[0, 3] = 5.0
        values = {
            "ctc_logits": np.zeros((1, 10, 20), dtype=np.float32),
            "acoustic_lengths": np.asarray([10], dtype=np.int64),
            "event_logits": event,
            "event_start_logits": event,
            "event_end_logits": event,
            "event_presence_logits": np.asarray(
                [[8.0, -8.0, -8.0, 8.0, -8.0, -8.0, -8.0]], dtype=np.float32
            ),
            "style_logits": style,
            "affect_logits": affect,
            "vad": np.asarray([[-0.5, 0.8, 0.4]], dtype=np.float32),
            "ood_logit": np.asarray([-8.0], dtype=np.float32),
            "ood_embedding": np.zeros((1, 4), dtype=np.float32),
        }
        return [values[name] for name in names]


def _wav(seconds: int = 1) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000 * seconds)
    return target.getvalue()


def _calibration(**changes) -> RuntimeCalibration:
    payload = {
        "event_thresholds": {
            label: 0.5
            for label in ("laugh", "sob", "scream", "sigh", "cough", "throat_clear", "sneeze")
        },
        "event_presence_thresholds": {
            label: 0.5
            for label in ("laugh", "sob", "scream", "sigh", "cough", "throat_clear", "sneeze")
        },
        "style_thresholds": {"shouting": 0.5, "whispering": 0.5},
        "vad_available": True,
        "ood_centroid": [0.0] * 4,
        "ood_distance_scale": 1.0,
        "ood_available": True,
    }
    payload.update(changes)
    return RuntimeCalibration.model_validate(payload)


def test_onnx_backend_emits_honest_scope_and_calibrated_affect() -> None:
    backend = OnnxAttuneBackend(
        Path("fixture-int8.onnx"),
        feature_extractor=lambda _wav: np.zeros((12, 80), dtype=np.float32),
        transcript_decoder=lambda _logits, _length: ("hello", 0.9),
        calibration=_calibration(
            localized_event_min_confidence=0.5,
            event_presence_enabled_labels=["sigh"],
            style_enabled_labels=["shouting"],
        ),
        session=FakeSession(),
    )
    result = backend.analyse_wav(_wav())

    assert result.events[0].label == "laugh"
    assert result.events[0].temporal_scope == "localized"
    assert result.events[1].label == "sigh"
    assert result.events[1].temporal_scope == "utterance"
    assert result.styles[0].label == "shouting"
    assert result.styles[0].temporal_scope == "utterance"
    assert result.styles[0].start_ms is None
    assert result.affect.top_label == "anger"
    assert result.affect.arousal.value > 0.7
    assert len(result.affect_spans) == 1
    assert result.transcript.words == []


def test_long_audio_receives_non_overlapping_affect_windows() -> None:
    backend = OnnxAttuneBackend(
        Path("fixture-int8.onnx"),
        feature_extractor=lambda _wav: np.zeros((108, 80), dtype=np.float32),
        transcript_decoder=lambda _logits, _length: ("hello", 0.9),
        calibration=_calibration(),
        session=FakeSession(),
    )

    result = backend.analyse_wav(_wav(9))

    assert [(span.start_ms, span.end_ms) for span in result.affect_spans] == [
        (0, 4_000),
        (4_000, 9_000),
    ]


def test_default_runtime_suppresses_unvalidated_presence_and_styles() -> None:
    backend = OnnxAttuneBackend(
        Path("fixture-int8.onnx"),
        feature_extractor=lambda _wav: np.zeros((12, 80), dtype=np.float32),
        transcript_decoder=lambda _logits, _length: ("hello", 0.9),
        calibration=_calibration(),
        session=FakeSession(),
    )

    result = backend.analyse_wav(_wav())

    assert len(result.events) == 1
    assert result.events[0].label == "laugh"
    assert result.events[0].temporal_scope == "localized"
    assert result.styles == []
    assert result.transcript.text == "hello"


def test_calibrated_probe_adds_utterance_annotation_without_raw_aed() -> None:
    class Output:
        name = PROBE_EMBEDDING_OUTPUT

    class ProbeSession(FakeSession):
        def get_outputs(self):
            return [Output()]

        def run(self, names, inputs):
            base = super().run(names[: len(OUTPUT_NAMES)], inputs)
            return [*base, np.zeros((1, 2), dtype=np.float32)]

    class Probe:
        def predict(self, embedding):
            assert embedding.shape == (2,)
            return ProbeDecision(
                channel="event",
                label="cough",
                confidence=0.91,
                abstained=False,
                source_label="cough",
                score=0.8,
                threshold=0.7,
            )

    backend = OnnxAttuneBackend(
        Path("fixture.onnx"),
        feature_extractor=lambda _wav: np.zeros((12, 80), dtype=np.float32),
        transcript_decoder=lambda _logits, _length: ("hello", 0.9),
        calibration=_calibration(localized_event_labels=[]),
        session=ProbeSession(),
        probe_heads=(Probe(),),
    )

    result = backend.analyse_wav(_wav())

    assert len(result.events) == 1
    assert result.events[0].label == "cough"
    assert result.events[0].temporal_scope == "utterance"


def test_missing_ood_calibration_forces_affect_abstention() -> None:
    backend = OnnxAttuneBackend(
        Path("fixture.onnx"),
        feature_extractor=lambda _wav: np.zeros((12, 80), dtype=np.float32),
        transcript_decoder=lambda _logits, _length: ("", 0.0),
        calibration=_calibration(ood_available=False),
        session=FakeSession(),
    )
    result = backend.analyse_wav(_wav())
    assert result.affect.abstain is True
    assert result.affect.abstention_reason == "out_of_distribution"
    assert result.affect.top_label is None


def test_untrained_vad_is_reported_unavailable() -> None:
    backend = OnnxAttuneBackend(
        Path("fixture.onnx"),
        feature_extractor=lambda _wav: np.zeros((12, 80), dtype=np.float32),
        transcript_decoder=lambda _logits, _length: ("", 0.0),
        calibration=_calibration(vad_available=False),
        session=FakeSession(),
    )
    result = backend.analyse_wav(_wav())
    assert result.affect.valence.available is False
    assert result.affect.valence.value is None


class Encoding:
    pieces = {1: " hello", 2: " world", 3: "!"}

    def decode(self, identifiers):
        return "".join(self.pieces[item] for item in identifiers)


class Tokenizer:
    encoding = Encoding()

    def decode(self, identifiers):
        return self.encoding.decode(identifiers)


def test_greedy_decoder_produces_ctc_derived_word_timestamps() -> None:
    logits = np.full((8, 5), -8.0, dtype=np.float32)
    logits[:, 0] = 2.0
    logits[1:3, 1] = 9.0
    logits[4:6, 2] = 9.0
    logits[6, 3] = 9.0
    decoder = GreedySenseVoiceDecoder(Tokenizer())

    text, confidence, words = decoder.decode_with_words(logits, 8, 800)

    assert text == "hello world!"
    assert confidence > 0.9
    assert [word["text"] for word in words] == ["hello", "world!"]
    assert words[0]["start_ms"] == 100
    assert words[1]["end_ms"] == 700


class SentencePieceProcessor:
    pieces = {1: "▁hello", 2: "▁world", 3: "!", 4: "▁", 5: "11"}

    def IdToPiece(self, identifier):
        return self.pieces[identifier]


class SentencePieceTokenizer:
    sp = SentencePieceProcessor()

    def decode(self, identifiers):
        return "".join(self.sp.IdToPiece(item) for item in identifiers).replace("▁", " ")


def test_greedy_decoder_uses_real_sensevoice_sentencepiece_boundaries() -> None:
    logits = np.full((8, 5), -8.0, dtype=np.float32)
    logits[:, 0] = 2.0
    logits[1:3, 1] = 9.0
    logits[4:6, 2] = 9.0
    logits[6, 3] = 9.0
    decoder = GreedySenseVoiceDecoder(SentencePieceTokenizer())

    text, confidence, words = decoder.decode_with_words(logits, 8, 800)

    assert text == "hello world!"
    assert confidence > 0.9
    assert [word["text"] for word in words] == ["hello", "world!"]
    assert words[0]["start_ms"] == 100
    assert words[1]["end_ms"] == 700


def test_greedy_decoder_honours_standalone_sentencepiece_boundary() -> None:
    logits = np.full((7, 7), -8.0, dtype=np.float32)
    logits[:, 0] = 2.0
    logits[1:3, 1] = 9.0
    logits[3, 4] = 9.0
    logits[4:6, 5] = 9.0
    decoder = GreedySenseVoiceDecoder(SentencePieceTokenizer())

    text, _confidence, words = decoder.decode_with_words(logits, 7, 700)

    assert text == "hello 11"
    assert [word["text"] for word in words] == ["hello", "11"]
