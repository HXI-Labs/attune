"""Frozen SenseVoiceSmall encoder embeddings for event probing."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from attune.models.frozen_event_probe import ProbeDataError

SENSEVOICE_EMBEDDING = "sensevoice-small-encoder-v1"
SENSEVOICE_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
QUERY_FRAMES = 4
TEMPORAL_BINS = 8


@contextmanager
def _offline_model_environment() -> Iterator[None]:
    """Keep local checkpoint loading from initiating model-hub requests."""
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenSenseVoiceEncoder:
    """Extract fixed-size embeddings while making encoder updates impossible."""

    output_size = 512 * (TEMPORAL_BINS + 2)

    def __init__(
        self,
        checkpoint: Path,
        cache_dir: Path,
        torch: Any,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if os.environ.get("ATTUNE_SENSEVOICE_LICENSE_REVIEWED") != "1":
            raise ProbeDataError(
                "SenseVoice weights require the reviewed model licence; set "
                "ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 only after reviewing "
                "data/provenance/sensevoice_small_weights.yaml"
            )
        checkpoint = checkpoint.resolve()
        model_file = checkpoint / "model.pt"
        if not model_file.is_file():
            raise ProbeDataError(f"SenseVoiceSmall model.pt is missing below {checkpoint}")

        if model_factory is None:
            try:
                from funasr import AutoModel
            except ImportError as error:
                raise ProbeDataError(
                    "SenseVoice extraction requires the model-runners dependencies"
                ) from error
            model_factory = AutoModel

        with _offline_model_environment():
            self.wrapper = model_factory(
                model=str(checkpoint),
                disable_update=True,
                device="cpu",
            )
        self.torch = torch
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir
        self.model_sha256 = file_sha256(model_file)
        self.cache_hits = 0
        self.cache_misses = 0

        self.model = self.wrapper.model
        self.model.eval()
        self.frontend = self.wrapper.kwargs["frontend"]
        self.frontend.eval()
        # FunASR's default 1.0 dither makes inference embeddings order-dependent.
        self.frontend.dither = 0.0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._assert_frozen()

    def _assert_frozen(self) -> None:
        trainable = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise ProbeDataError(
                "SenseVoice encoder is not frozen; trainable parameters include "
                + ", ".join(trainable[:5])
            )

    def _cache_path(self, audio_path: Path) -> Path:
        digest = hashlib.sha256()
        digest.update(SENSEVOICE_EMBEDDING.encode())
        digest.update(b"frontend-dither=0")
        digest.update(self.model_sha256.encode())
        digest.update(file_sha256(audio_path).encode())
        return self.cache_dir / f"{digest.hexdigest()}.pt"

    def __call__(self, audio_path: Path) -> Any:
        """Return pooled acoustic encoder frames for one audio clip."""
        cache_path = self._cache_path(audio_path)
        if cache_path.is_file():
            embedding = self.torch.load(cache_path, map_location="cpu", weights_only=True)
            if tuple(embedding.shape) != (self.output_size,):
                raise ProbeDataError(f"invalid cached embedding shape in {cache_path}")
            self.cache_hits += 1
            return embedding

        captured: list[Any] = []

        def capture_encoder_output(_module: Any, _inputs: Any, output: Any) -> None:
            captured.append(output)

        hook = self.model.encoder.register_forward_hook(capture_encoder_output)
        try:
            self._assert_frozen()
            with self.torch.inference_mode(), _offline_model_environment():
                self.wrapper.generate(input=str(audio_path), cache={}, language="auto")
        finally:
            hook.remove()

        if len(captured) != 1 or not isinstance(captured[0], tuple):
            raise ProbeDataError(
                f"SenseVoice encoder extraction produced {len(captured)} unexpected outputs"
            )
        encoder_out, encoder_lengths = captured[0][:2]
        length = int(encoder_lengths[0].item())
        if length <= QUERY_FRAMES:
            raise ProbeDataError(f"SenseVoice encoder returned no acoustic frames for {audio_path}")
        acoustic = encoder_out[0, QUERY_FRAMES:length].detach().float().cpu()
        if acoustic.ndim != 2 or acoustic.shape[1] != 512:
            raise ProbeDataError(
                f"unexpected SenseVoice encoder shape {tuple(acoustic.shape)} for {audio_path}"
            )

        temporal = self.torch.nn.functional.adaptive_avg_pool1d(
            acoustic.transpose(0, 1).unsqueeze(0),
            TEMPORAL_BINS,
        ).flatten()
        embedding = self.torch.cat(
            (temporal, acoustic.mean(dim=0), acoustic.std(dim=0, unbiased=False))
        )
        self._assert_frozen()
        if embedding.requires_grad or embedding.grad_fn is not None:
            raise ProbeDataError("SenseVoice embedding unexpectedly retained an autograd graph")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(embedding, cache_path)
        self.cache_misses += 1
        return embedding

    def metadata(self) -> dict[str, Any]:
        """Return non-weight provenance and freeze evidence for the metrics report."""
        return {
            "name": SENSEVOICE_EMBEDDING,
            "model": "SenseVoiceSmall by FunASR/FunAudioLLM",
            "revision": SENSEVOICE_REVISION,
            "model_file_sha256": self.model_sha256,
            "trainable_parameters": 0,
            "total_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "query_frames_excluded": QUERY_FRAMES,
            "frontend_dither": self.frontend.dither,
            "pooling": f"{TEMPORAL_BINS} temporal means plus acoustic-frame mean/std",
            "output_size": self.output_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
