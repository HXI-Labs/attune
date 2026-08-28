"""Last SenseVoice encoder block plus frame MLP: not a frozen encoder."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from attune.models.frozen_event_probe import ProbeDataError
from attune.models.sensevoice_probe import (
    FRAME_SIZE,
    QUERY_FRAMES,
    SENSEVOICE_REVISION,
    _load_fbank,
    _offline_model_environment,
    file_sha256,
)

SENSEVOICE_LAST_BLOCK_FRAMES = "sensevoice-small-encoder-lastblock-frames-v1"
PREFIX_CACHE_TAG = b"last-block-prefix-v1"


class LastBlockSenseVoiceFrameEncoder:
    """Unfreeze only ``encoder.tp_encoders[-1]``; keep the frozen extraction route."""

    output_size = FRAME_SIZE

    def __init__(
        self,
        checkpoint: Path,
        cache_dir: Path,
        torch: Any,
        *,
        model_factory: Callable[..., Any] | None = None,
        feature_loader: Callable[[Path, Any, Any], tuple[Any, Any]] | None = None,
        prefix_cache: bool = True,
    ) -> None:
        import os

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
        self.prefix_cache = prefix_cache
        self.model_sha256 = file_sha256(model_file)
        self.cache_hits = 0
        self.cache_misses = 0
        self.feature_loader = feature_loader or _load_fbank

        self.model = self.wrapper.model
        self.model.eval()
        self.frontend = self.wrapper.kwargs["frontend"]
        self.frontend.eval()
        self.frontend.dither = 0.0
        self._require_last_block()
        self._freeze_except_last_block()
        self._assert_last_block_only_trainable()

    def _require_last_block(self) -> None:
        encoder = getattr(self.model, "encoder", None)
        blocks = getattr(encoder, "tp_encoders", None)
        if encoder is None or blocks is None or len(blocks) < 1:
            raise ProbeDataError(
                "SenseVoice encoder is missing tp_encoders; cannot unfreeze a last block"
            )
        if not hasattr(encoder, "tp_norm"):
            raise ProbeDataError("SenseVoice encoder is missing tp_norm after the last SANM block")

    @property
    def last_block(self) -> Any:
        return self.model.encoder.tp_encoders[-1]

    @property
    def last_block_index(self) -> int:
        return len(self.model.encoder.tp_encoders) - 1

    @property
    def last_block_name(self) -> str:
        return f"encoder.tp_encoders.{self.last_block_index}"

    def _freeze_except_last_block(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        frontend_parameters = getattr(self.frontend, "parameters", None)
        if callable(frontend_parameters):
            for parameter in frontend_parameters():
                parameter.requires_grad_(False)
        for parameter in self.last_block.parameters():
            parameter.requires_grad_(True)

    def _assert_last_block_only_trainable(self) -> None:
        prefix = self.last_block_name + "."
        trainable = [
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ProbeDataError("last SenseVoice encoder block has no trainable parameters")
        illegal = [name for name in trainable if not name.startswith(prefix)]
        if illegal:
            raise ProbeDataError(
                "trainable parameters outside the last encoder block: " + ", ".join(illegal[:8])
            )

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )

    def prepare_last_block(self, *, train: bool) -> None:
        """Keep frozen modules in eval; toggle dropout only on the last SANM block."""
        self.model.eval()
        self.frontend.eval()
        if train:
            self.last_block.train()
        else:
            self.last_block.eval()
        self._assert_last_block_only_trainable()

    def _prefix_cache_path(self, audio_path: Path) -> Path:
        digest = hashlib.sha256()
        digest.update(SENSEVOICE_LAST_BLOCK_FRAMES.encode())
        digest.update(PREFIX_CACHE_TAG)
        digest.update(b"frontend-dither=0")
        digest.update(b"query-frames-kept-in-prefix=4")
        digest.update(self.last_block_name.encode())
        digest.update(self.model_sha256.encode())
        digest.update(file_sha256(audio_path).encode())
        return self.cache_dir / "prefix" / f"{digest.hexdigest()}.pt"

    def _prepare_encoder_input(self, audio_path: Path) -> tuple[Any, Any]:
        """Match FrozenSenseVoiceEncoder: dither-0 frontend, then four query embeddings."""
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
        speech_lengths = speech_lengths + 1
        event_emo_query = self.model.embed(self.torch.LongTensor([[1, 2]])).repeat(
            speech.size(0), 1, 1
        )
        input_query = self.torch.cat((language_query, event_emo_query), dim=1)
        speech = self.torch.cat((input_query, speech), dim=1)
        speech_lengths = speech_lengths + 3
        return speech, speech_lengths

    def prefix_for(self, audio_path: Path) -> tuple[Any, Any]:
        """Return detached last-block input ``(x, mask)``. Safe to cache; frozen."""
        cache_path = self._prefix_cache_path(audio_path)
        if self.prefix_cache and cache_path.is_file():
            payload = self.torch.load(cache_path, map_location="cpu", weights_only=True)
            self.cache_hits += 1
            return payload["x"], payload["mask"]

        captured: dict[str, Any] = {}

        def _hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            captured["x"] = inputs[0]
            captured["mask"] = inputs[1] if len(inputs) > 1 else None

        handle = self.last_block.register_forward_pre_hook(_hook)
        try:
            # no_grad, not inference_mode: captured tensors must be replayable later.
            with self.torch.no_grad(), _offline_model_environment():
                self.model.eval()
                self.last_block.eval()
                speech, speech_lengths = self._prepare_encoder_input(audio_path)
                self.model.encoder(speech, speech_lengths)
        finally:
            handle.remove()
        if "x" not in captured:
            raise ProbeDataError(f"failed to capture last-block prefix for {audio_path}")
        prefix = captured["x"].detach().float().cpu().contiguous()
        mask = captured["mask"]
        if mask is not None:
            mask = mask.detach().cpu().contiguous()
        if self.prefix_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.torch.save({"x": prefix, "mask": mask}, cache_path)
        self.cache_misses += 1
        return prefix, mask

    def frames_from_prefix(self, prefix: Any, mask: Any, *, train: bool) -> Any:
        """Run only the last SANM block + frozen ``tp_norm``, then strip query frames."""
        self.prepare_last_block(train=train)
        prefix = prefix.detach()
        if mask is not None:
            mask = mask.detach()
        outputs = self.last_block(prefix, mask)
        encoded = outputs[0] if isinstance(outputs, tuple) else outputs
        encoded = self.model.encoder.tp_norm(encoded)
        if mask is None:
            length = int(encoded.size(1))
        else:
            length = int(mask.reshape(mask.size(0), -1)[0].sum().item())
        if length <= QUERY_FRAMES:
            raise ProbeDataError("SenseVoice encoder returned no acoustic frames")
        acoustic = encoded[0, QUERY_FRAMES:length].float()
        if acoustic.ndim != 2 or acoustic.shape[1] != FRAME_SIZE:
            raise ProbeDataError(f"unexpected last-block frame shape {tuple(acoustic.shape)}")
        if train:
            if not acoustic.requires_grad:
                raise ProbeDataError("last-block acoustic frames did not retain an autograd graph")
            return acoustic
        return acoustic.detach()

    def __call__(self, audio_path: Path, *, train: bool = False) -> Any:
        prefix, mask = self.prefix_for(audio_path)
        return self.frames_from_prefix(prefix, mask, train=train)

    @property
    def frame_hop_ms(self) -> float:
        return float(getattr(self.frontend, "frame_shift", 10)) * float(
            getattr(self.frontend, "lfr_n", 6)
        )

    @property
    def first_frame_center_ms(self) -> float:
        return self.frame_hop_ms / 2

    def frame_centers_ms(self, frame_count: int) -> tuple[float, ...]:
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        return tuple(
            self.first_frame_center_ms + index * self.frame_hop_ms for index in range(frame_count)
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": SENSEVOICE_LAST_BLOCK_FRAMES,
            "model": "SenseVoiceSmall by FunASR/FunAudioLLM",
            "revision": SENSEVOICE_REVISION,
            "model_file_sha256": self.model_sha256,
            "last_block_name": self.last_block_name,
            "trainable_parameters": self.trainable_parameter_count(),
            "trainable_parameter_names_head": list(self.trainable_parameter_names()[:8]),
            "total_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "query_frames_excluded": QUERY_FRAMES,
            "frontend_dither": self.frontend.dither,
            "extraction_route": "direct_frontend_and_last_block_encoder",
            "prefix_cache": self.prefix_cache,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "output_shape": ["T", FRAME_SIZE],
            "pooling": None,
            "frame_hop_ms_approx": self.frame_hop_ms,
            "first_acoustic_frame_center_ms_approx": self.first_frame_center_ms,
            "query_prefix_time_semantics": (
                "four non-acoustic query positions are removed; they do not shift audio time"
            ),
        }
