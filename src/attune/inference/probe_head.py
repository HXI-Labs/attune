"""Torch-free inference for calibrated linear heads over Cadence embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from attune.models.sensevoice_probe import SENSEVOICE_EN_EMBEDDING

ProbeChannel = Literal["event", "style"]


@dataclass(frozen=True)
class ProbeDecision:
    channel: ProbeChannel
    label: str
    confidence: float
    abstained: bool
    source_label: str
    score: float
    threshold: float


class LinearProbeHead:
    """Validated NumPy artifact produced from a frozen-encoder probe checkpoint."""

    def __init__(self, artifact: Path) -> None:
        with np.load(artifact, allow_pickle=False) as values:
            self.format = str(values["format"].item())
            self.embedding_name = str(values["embedding"].item())
            self.labels = tuple(str(value) for value in values["labels"].tolist())
            self.channels = tuple(str(value) for value in values["channels"].tolist())
            self.target_labels = tuple(str(value) for value in values["target_labels"].tolist())
            self.weight = values["weight"].astype(np.float32)
            self.bias = values["bias"].astype(np.float32)
            self.feature_mean = values["feature_mean"].astype(np.float32)
            self.feature_scale = values["feature_scale"].astype(np.float32)
            self.abstention_method = str(values["abstention_method"].item())
            self.abstention_threshold = float(values["abstention_threshold"].item())
            self.temperature = float(values["temperature"].item())
        self._validate(artifact)

    def _validate(self, artifact: Path) -> None:
        class_count = len(self.labels)
        feature_count = self.feature_mean.size
        expected_outputs = class_count + (self.abstention_method == "none_logit")
        if self.format != "attune_linear_probe_v1":
            raise ValueError(f"unsupported probe artifact format in {artifact}")
        if self.embedding_name != SENSEVOICE_EN_EMBEDDING:
            raise ValueError(f"compact ONNX probes require {SENSEVOICE_EN_EMBEDDING}: {artifact}")
        if not class_count or len(self.channels) != class_count:
            raise ValueError(f"incomplete probe labels in {artifact}")
        if len(self.target_labels) != class_count:
            raise ValueError(f"incomplete probe target labels in {artifact}")
        if any(channel not in {"event", "style"} for channel in self.channels):
            raise ValueError(f"unsupported probe channel in {artifact}")
        if self.abstention_method != "none_logit":
            raise ValueError(f"release probes require none-logit abstention: {artifact}")
        if self.weight.shape != (expected_outputs, feature_count):
            raise ValueError(f"probe weight shape does not match labels in {artifact}")
        if self.bias.shape != (expected_outputs,):
            raise ValueError(f"probe bias shape does not match labels in {artifact}")
        if self.feature_scale.shape != (feature_count,):
            raise ValueError(f"probe feature scale is invalid in {artifact}")
        if np.any(self.feature_scale <= 0):
            raise ValueError(f"probe feature scale must be positive in {artifact}")
        if not np.isfinite(self.abstention_threshold) or self.temperature <= 0:
            raise ValueError(f"invalid probe calibration in {artifact}")

    @property
    def parameter_count(self) -> int:
        return int(self.weight.size + self.bias.size)

    def predict(self, embedding: np.ndarray) -> ProbeDecision:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != self.feature_mean.shape:
            raise ValueError(
                f"probe embedding has shape {vector.shape}, expected {self.feature_mean.shape}"
            )
        normalized = (vector - self.feature_mean) / self.feature_scale
        logits = self.weight @ normalized + self.bias
        raw_probabilities = _softmax(logits)
        class_count = len(self.labels)
        source_index = int(raw_probabilities[:class_count].argmax())
        score = float(raw_probabilities[source_index] - raw_probabilities[class_count])
        calibrated = _softmax(logits / self.temperature)
        return ProbeDecision(
            channel=self.channels[source_index],
            label=self.target_labels[source_index],
            confidence=float(calibrated[source_index]),
            abstained=score < self.abstention_threshold,
            source_label=self.labels[source_index],
            score=score,
            threshold=self.abstention_threshold,
        )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exponential = np.exp(shifted)
    return exponential / exponential.sum()
