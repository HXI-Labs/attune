"""Inference for the compact, truncated emotion2vec+ affect student."""

from __future__ import annotations

import array
import hashlib
import io
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from attune.models.truncated_emotion2vec import TruncatedEmotion2VecHead, pool_layer
from attune.schema.output import AffectCategory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_source_checkpoint(path: Path, expected_sha256: str) -> bool:
    if _sha256(path) == expected_sha256:
        return True
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return bool(
        isinstance(checkpoint, dict)
        and checkpoint.get("attune_source_checkpoint_sha256") == expected_sha256
    )


def _waveform(payload: bytes) -> torch.Tensor:
    try:
        with wave.open(io.BytesIO(payload), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != 16_000
            ):
                raise ValueError("emotion2vec student requires 16 kHz mono PCM16 WAV audio")
            samples = array.array("h", audio.readframes(audio.getnframes()))
    except (EOFError, wave.Error) as error:
        raise ValueError(f"invalid WAV payload: {error}") from error
    if not samples:
        raise ValueError("audio payload contains no samples")
    waveform = torch.tensor(samples, dtype=torch.float32) / 32_768
    return functional.layer_norm(waveform, waveform.shape)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TruncatedEmotion2VecPredictor:
    """Run a frozen emotion2vec+ frontend, early blocks, and the trained affect head."""

    def __init__(
        self,
        teacher_path: Path,
        checkpoint_path: Path,
        *,
        device: str = "auto",
        batch_size: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("emotion2vec batch size must be positive")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != "1.0":
            raise ValueError("invalid truncated emotion2vec checkpoint")
        labels = tuple(checkpoint.get("labels", ()))
        expected_labels = tuple(category.value for category in AffectCategory)
        if labels != expected_labels:
            raise ValueError("truncated emotion2vec checkpoint has incompatible affect labels")
        depth = int(checkpoint["depth"])
        if depth < 1 or depth > 3:
            raise ValueError("truncated emotion2vec checkpoint depth must be between 1 and 3")
        teacher_checkpoint = teacher_path / "model.pt"
        if not _matches_source_checkpoint(
            teacher_checkpoint,
            str(checkpoint["teacher_checkpoint_sha256"]),
        ):
            raise ValueError("emotion2vec+ teacher checkpoint hash mismatch")

        from funasr import AutoModel

        wrapper = AutoModel(model=str(teacher_path), disable_update=True)
        model = wrapper.model.eval()
        model.blocks = nn.ModuleList(list(model.blocks)[:depth])
        model.modality_encoders["AUDIO"].decoder = nn.Identity()
        model.proj = nn.Identity()
        self.device = _device(device)
        self.model = model.to(self.device)
        feature_mean = checkpoint["feature_mean"].float()
        self.feature_mean = feature_mean.to(self.device)
        self.feature_scale = checkpoint["feature_scale"].float().to(self.device)
        head = TruncatedEmotion2VecHead(
            feature_mean.numel(),
            len(labels),
            kind=str(checkpoint.get("head_kind", "mlp")),
        )
        head.load_state_dict(checkpoint["state_dict"])
        self.head = head.eval().to(self.device)
        self.depth = depth
        self.temperature = float(checkpoint["temperature"])
        self.labels = labels
        self.parameter_count = int(checkpoint["truncated_emotion2vec_parameters"])
        self.batch_size = batch_size

    def predict_probabilities(self, payloads: Sequence[bytes]) -> np.ndarray:
        if not payloads:
            return np.empty((0, len(self.labels)), dtype=np.float32)
        predictions = []
        for start in range(0, len(payloads), self.batch_size):
            waveforms = [
                _waveform(payload) for payload in payloads[start : start + self.batch_size]
            ]
            lengths = torch.tensor([waveform.numel() for waveform in waveforms])
            samples = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
            padding_mask = torch.arange(samples.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
            with torch.inference_mode():
                extracted = self.model.extract_features(
                    samples.to(self.device),
                    padding_mask=padding_mask.to(self.device),
                )
                output_mask = extracted["padding_mask"]
                if output_mask is None:
                    output_mask = torch.zeros(
                        extracted["x"].shape[:2],
                        dtype=torch.bool,
                        device=self.device,
                    )
                features = pool_layer(extracted["layer_results"][self.depth - 1], output_mask)
                normalized = (features - self.feature_mean) / self.feature_scale
                logits = self.head(normalized)
                predictions.append((logits / self.temperature).softmax(dim=-1).cpu().numpy())
        return np.concatenate(predictions, axis=0)
