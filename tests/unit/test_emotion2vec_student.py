import io
import json
import sys
import types
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from attune.inference.emotion2vec_export import _convert_padding_mask
from attune.inference.emotion2vec_student import (
    OnnxTruncatedEmotion2VecPredictor,
    TorchScriptTruncatedEmotion2VecPredictor,
    TruncatedEmotion2VecPredictor,
)
from attune.schema.output import AffectCategory


def test_predictor_rejects_unknown_quantization_before_loading_assets() -> None:
    with pytest.raises(ValueError, match="quantization must be fp32 or int8"):
        TruncatedEmotion2VecPredictor(
            Path("missing-teacher"),
            Path("missing-checkpoint"),
            quantization="int4",
        )


def _wav(sample_count: int) -> bytes:
    output = io.BytesIO()
    samples = (np.arange(sample_count, dtype=np.int16) % 200).tobytes()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(samples)
    return output.getvalue()


def test_padding_conversion_uses_each_unpadded_length() -> None:
    frontend = types.SimpleNamespace(feature_enc_layers=[(1, 3, 2)])
    input_mask = torch.tensor(
        [
            [False] * 10,
            [False] * 7 + [True] * 3,
        ]
    )
    features = torch.zeros((2, 4, 3))

    converted = _convert_padding_mask(frontend, features, input_mask)

    assert converted.tolist() == [
        [False, False, False, False],
        [False, False, False, True],
    ]


def test_onnx_predictor_batches_waveforms_and_reads_model_metadata(monkeypatch, tmp_path) -> None:
    labels = tuple(category.value for category in AffectCategory)

    class FakeSession:
        def __init__(self, _path, providers):
            assert providers == ["CPUExecutionProvider"]
            self.feeds = []

        def get_modelmeta(self):
            return types.SimpleNamespace(
                custom_metadata_map={
                    "attune.schema_version": "1.0",
                    "attune.labels": json.dumps(labels),
                    "attune.parameter_count": "57937364",
                    "attune.quantization": "int8",
                }
            )

        def get_inputs(self):
            return [
                types.SimpleNamespace(name="samples"),
                types.SimpleNamespace(name="padding_mask"),
            ]

        def run(self, _outputs, feed):
            self.feeds.append(feed)
            probabilities = np.zeros((feed["samples"].shape[0], len(labels)), dtype=np.float32)
            probabilities[:, 0] = 1.0
            return [probabilities]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(InferenceSession=FakeSession),
    )
    model_path = tmp_path / "affect-int8.onnx"
    model_path.write_bytes(b"fake ONNX artifact")
    predictor = OnnxTruncatedEmotion2VecPredictor(model_path, batch_size=2)

    probabilities = predictor.predict_probabilities([_wav(400), _wav(320)])

    assert probabilities.shape == (2, len(labels))
    assert predictor.parameter_count == 57_937_364
    assert predictor.quantization == "int8"
    feed = predictor.session.feeds[0]
    assert feed["samples"].shape == (2, 400)
    assert not feed["padding_mask"][0].any()
    assert feed["padding_mask"][1, 320:].all()


def test_torchscript_predictor_reads_embedded_metadata(monkeypatch, tmp_path) -> None:
    labels = tuple(category.value for category in AffectCategory)

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, samples, _padding_mask):
            probabilities = torch.zeros((samples.shape[0], len(labels)))
            probabilities[:, 0] = 1.0
            return probabilities

    def fake_load(_path, *, map_location):
        assert map_location == "cpu"
        return FakeModel()

    model_path = tmp_path / "affect-int8.pt"
    with zipfile.ZipFile(model_path, "w") as archive:
        archive.writestr(
            "affect-int8/extra/attune.json",
            json.dumps(
                {
                    "schema_version": "1.0",
                    "labels": labels,
                    "parameter_count": 57_937_364,
                    "quantization": "int8",
                    "quantization_engine": "qnnpack",
                }
            ),
        )

    monkeypatch.setattr(torch.jit, "load", fake_load)
    predictor = TorchScriptTruncatedEmotion2VecPredictor(
        model_path,
        batch_size=2,
    )

    probabilities = predictor.predict_probabilities([_wav(400), _wav(320)])

    assert probabilities.shape == (2, len(labels))
    assert probabilities[:, 0].tolist() == [1.0, 1.0]
    assert predictor.quantization == "int8"
