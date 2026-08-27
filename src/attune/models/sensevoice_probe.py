"""Frozen SenseVoiceSmall encoder embeddings for event probing."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from attune.models.frozen_event_probe import ProbeDataError

SENSEVOICE_EMBEDDING = "sensevoice-small-encoder-v2"
SENSEVOICE_FRAME_EMBEDDING = "sensevoice-small-encoder-frames-v1"
SENSEVOICE_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
QUERY_FRAMES = 4
TEMPORAL_BINS = 8
FRAME_SIZE = 512


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

    output_size = FRAME_SIZE * (TEMPORAL_BINS + 2)

    def __init__(
        self,
        checkpoint: Path,
        cache_dir: Path,
        torch: Any,
        *,
        model_factory: Callable[..., Any] | None = None,
        feature_loader: Callable[[Path, Any, Any], tuple[Any, Any]] | None = None,
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
        self.feature_loader = feature_loader or _load_fbank

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
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
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
        digest.update(b"direct-encoder-v1")
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

        acoustic = self._extract_acoustic_frames(audio_path)
        embedding = pool_sensevoice_frames(acoustic, self.torch)
        self._assert_frozen()
        if embedding.requires_grad or embedding.grad_fn is not None:
            raise ProbeDataError("SenseVoice embedding unexpectedly retained an autograd graph")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(embedding, cache_path)
        self.cache_misses += 1
        return embedding

    def _extract_acoustic_frames(self, audio_path: Path) -> Any:
        """Run the deterministic frontend and frozen encoder without generation."""
        self._assert_frozen()
        with self.torch.inference_mode(), _offline_model_environment():
            speech, speech_lengths = self.feature_loader(audio_path, self.frontend, self.torch)
            speech = speech.to(device="cpu")
            speech_lengths = speech_lengths.to(device="cpu")

            language_query = self.model.embed(
                self.torch.LongTensor([[self.model.lid_dict["auto"]]])
            ).repeat(speech.size(0), 1, 1)
            textnorm_query = self.model.embed(
                self.torch.LongTensor([[self.model.textnorm_dict["woitn"]]])
            ).repeat(speech.size(0), 1, 1)
            speech = self.torch.cat((textnorm_query, speech), dim=1)
            speech_lengths += 1
            event_emo_query = self.model.embed(self.torch.LongTensor([[1, 2]])).repeat(
                speech.size(0), 1, 1
            )
            input_query = self.torch.cat((language_query, event_emo_query), dim=1)
            speech = self.torch.cat((input_query, speech), dim=1)
            speech_lengths += 3
            encoder_out, encoder_lengths = self.model.encoder(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]

        length = int(encoder_lengths[0].item())
        if length <= QUERY_FRAMES:
            raise ProbeDataError(f"SenseVoice encoder returned no acoustic frames for {audio_path}")
        acoustic = encoder_out[0, QUERY_FRAMES:length].detach().float().cpu()
        if acoustic.ndim != 2 or acoustic.shape[1] != FRAME_SIZE:
            raise ProbeDataError(
                f"unexpected SenseVoice encoder shape {tuple(acoustic.shape)} for {audio_path}"
            )
        if acoustic.requires_grad or acoustic.grad_fn is not None:
            raise ProbeDataError("SenseVoice frames unexpectedly retained an autograd graph")
        self._assert_frozen()
        return acoustic

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
            "extraction_route": "direct_frontend_and_frozen_encoder",
            "pooling": f"{TEMPORAL_BINS} temporal means plus acoustic-frame mean/std",
            "output_size": self.output_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


class FrozenSenseVoiceFrameEncoder(FrozenSenseVoiceEncoder):
    """Return unpooled acoustic frames from the same immutable encoder."""

    output_size = FRAME_SIZE

    def _cache_path(self, audio_path: Path) -> Path:
        digest = hashlib.sha256()
        digest.update(SENSEVOICE_FRAME_EMBEDDING.encode())
        digest.update(b"frontend-dither=0")
        digest.update(b"direct-encoder-v1")
        digest.update(b"query-frames-stripped=4")
        digest.update(self.model_sha256.encode())
        digest.update(file_sha256(audio_path).encode())
        return self.cache_dir / SENSEVOICE_FRAME_EMBEDDING / f"{digest.hexdigest()}.pt"

    def __call__(self, audio_path: Path) -> Any:
        """Return a variable-length ``(T, 512)`` acoustic frame tensor."""
        cache_path = self._cache_path(audio_path)
        if cache_path.is_file():
            frames = self.torch.load(cache_path, map_location="cpu", weights_only=True)
            if frames.ndim != 2 or frames.shape[0] < 1 or frames.shape[1] != self.output_size:
                raise ProbeDataError(f"invalid cached frame shape in {cache_path}")
            self.cache_hits += 1
            return frames

        frames = self._extract_acoustic_frames(audio_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(frames, cache_path)
        self.cache_misses += 1
        return frames

    @property
    def frame_hop_ms(self) -> float:
        """Approximate LFR output hop in milliseconds."""
        return float(getattr(self.frontend, "frame_shift", 10)) * float(
            getattr(self.frontend, "lfr_n", 6)
        )

    @property
    def first_frame_center_ms(self) -> float:
        """Approximate centre of the first acoustic frontend frame."""
        return float(getattr(self.frontend, "frame_length", 25)) / 2

    def frame_centers_ms(self, frame_count: int) -> tuple[float, ...]:
        """Return approximate audio-time centres after query-frame removal."""
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        return tuple(
            self.first_frame_center_ms + index * self.frame_hop_ms for index in range(frame_count)
        )

    def metadata(self) -> dict[str, Any]:
        """Return frame extraction provenance and approximate time geometry."""
        metadata = super().metadata()
        metadata.update(
            {
                "name": SENSEVOICE_FRAME_EMBEDDING,
                "output_shape": ["T", FRAME_SIZE],
                "pooling": None,
                "frame_hop_ms_approx": self.frame_hop_ms,
                "first_acoustic_frame_center_ms_approx": self.first_frame_center_ms,
                "query_prefix_time_semantics": (
                    "four non-acoustic query positions are removed; they do not shift audio time"
                ),
            }
        )
        metadata.pop("output_size", None)
        return metadata


def pool_sensevoice_frames(acoustic: Any, torch: Any) -> Any:
    """Reconstruct the stable 5120-d probe embedding from acoustic frames."""
    if acoustic.ndim != 2 or acoustic.shape[0] < 1 or acoustic.shape[1] != FRAME_SIZE:
        raise ProbeDataError(
            f"expected non-empty (T, {FRAME_SIZE}) SenseVoice frames, got {tuple(acoustic.shape)}"
        )
    temporal = torch.nn.functional.adaptive_avg_pool1d(
        acoustic.transpose(0, 1).unsqueeze(0),
        TEMPORAL_BINS,
    ).flatten()
    return torch.cat((temporal, acoustic.mean(dim=0), acoustic.std(dim=0, unbiased=False)))


def _load_fbank(audio_path: Path, frontend: Any, torch: Any) -> tuple[Any, Any]:
    """Load one clip and run only the official deterministic FunASR frontend."""
    try:
        from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
    except ImportError as error:
        raise ProbeDataError(
            "SenseVoice extraction requires the model-runners dependencies"
        ) from error
    audio = load_audio_text_image_video(
        str(audio_path),
        fs=frontend.fs,
        audio_fs=16_000,
        data_type="sound",
    )
    speech, speech_lengths = extract_fbank(audio, data_type="sound", frontend=frontend)
    return speech.to(torch.float32), speech_lengths.to(torch.int32)
