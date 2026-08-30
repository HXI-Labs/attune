"""Budget-aware trainer for the unified Attune candidate."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from attune.models.joint import (
    AttuneJointModel,
    attune_delta_checkpoint,
    load_attune_checkpoint,
)
from attune.training.data import JointFeatureDataset, collate_joint_examples, manifest_sha256
from attune.training.losses import JointTargets, LossWeights, compute_joint_loss

_TRAINING_SOURCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "scripts/train_joint.py",
    "src/attune/models/joint.py",
    "src/attune/training/data.py",
    "src/attune/training/losses.py",
    "src/attune/training/trainer.py",
)
_TARGET_PARAMETER_PREFIXES = {
    "affect": ("affect_projection.", "affect_head."),
    "events": ("event_head.",),
    "styles": ("style_projection.", "style_head."),
}


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 4
    samples_per_epoch: int | None = None
    validation_batch_size: int | None = None
    gradient_accumulation: int = 1
    head_learning_rate: float = 1e-4
    encoder_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"
    maximum_cost_gbp: float = 95.0
    gpu_hour_cost_gbp: float = 0.36
    early_stopping_patience: int = 3
    corpus_balancing: bool = True
    resume: bool = True
    log_interval_steps: int = 25
    duration_bucket_multiplier: int = 20
    include_ctc_loss: bool = True
    training_target: str | None = None

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("epochs, batch_size, and gradient_accumulation must be positive")
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive when set")
        if self.validation_batch_size is not None and self.validation_batch_size < 1:
            raise ValueError("validation_batch_size must be positive when set")
        if self.log_interval_steps < 1:
            raise ValueError("log_interval_steps must be positive")
        if self.duration_bucket_multiplier < 1:
            raise ValueError("duration_bucket_multiplier must be positive")
        if self.maximum_cost_gbp <= 0 or self.gpu_hour_cost_gbp < 0:
            raise ValueError("compute budget values must be non-negative")
        if (
            self.training_target is not None
            and self.training_target not in JointFeatureDataset._TARGET_FIELDS
        ):
            raise ValueError(f"unsupported training target: {self.training_target}")


def _parameter_groups(model: AttuneJointModel, config: TrainerConfig) -> list[dict[str, Any]]:
    encoder_ids = {
        id(parameter) for parameter in model.sensevoice.parameters() if parameter.requires_grad
    }
    encoder = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in encoder_ids
    ]
    heads = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    groups = [{"params": heads, "lr": config.head_learning_rate}]
    if encoder:
        groups.append({"params": encoder, "lr": config.encoder_learning_rate})
    return groups


def _estimated_cost(started: float, hourly_cost: float) -> float:
    return (time.monotonic() - started) / 3600.0 * hourly_cost


def _targets_for_loss(targets: JointTargets, *, include_ctc_loss: bool) -> JointTargets:
    """Exclude CTC targets when the ASR route cannot be changed by training."""

    if include_ctc_loss:
        return targets
    return replace(
        targets,
        ctc_targets=None,
        ctc_target_lengths=None,
        ctc_example_mask=None,
    )


def _restrict_training_target(model: AttuneJointModel, target: str | None) -> None:
    prefixes = _TARGET_PARAMETER_PREFIXES.get(target)
    if prefixes is None:
        return
    for name, parameter in model.named_parameters():
        if not name.startswith(prefixes):
            parameter.requires_grad_(False)


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json_write(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _training_source_sha256() -> str:
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for relative in _TRAINING_SOURCE_FILES:
        path = root / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _corpus_balanced_sampler(
    dataset: JointFeatureDataset,
    *,
    generator: torch.Generator,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Sample corpora uniformly while retaining natural variation within each corpus."""

    counts: dict[str, int] = {}
    for row in dataset.rows:
        counts[row.dataset_id] = counts.get(row.dataset_id, 0) + 1
    weights = torch.tensor(
        [1.0 / counts[row.dataset_id] for row in dataset.rows], dtype=torch.double
    )
    return WeightedRandomSampler(
        weights,
        num_samples=num_samples or len(dataset),
        replacement=True,
        generator=generator,
    )


class DurationBucketBatchSampler(Sampler[list[int]]):
    """Batch nearby durations from shuffled sampling windows to limit padding."""

    def __init__(
        self,
        sampler: Sampler[int],
        durations_ms: list[int],
        *,
        batch_size: int,
        bucket_multiplier: int,
    ) -> None:
        self.sampler = sampler
        self.durations_ms = durations_ms
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_multiplier

    def __iter__(self):
        bucket: list[int] = []
        for index in self.sampler:
            bucket.append(index)
            if len(bucket) == self.bucket_size:
                yield from self._batches(bucket)
                bucket = []
        if bucket:
            yield from self._batches(bucket)

    def _batches(self, indices: list[int]):
        indices.sort(key=self.durations_ms.__getitem__)
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.sampler) / self.batch_size)


def train_joint_model(
    model: AttuneJointModel,
    *,
    manifest: Path,
    output_dir: Path,
    config: TrainerConfig,
    weights: LossWeights | None = None,
    initial_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    config.validate()
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    device = torch.device(config.device)
    if config.encoder_learning_rate == 0:
        for parameter in model.sensevoice.parameters():
            parameter.requires_grad_(False)
    _restrict_training_target(model, config.training_target)
    model.to(device)
    parameter_summary = model.trainable_parameter_summary()
    if (
        not config.include_ctc_loss
        and parameter_summary["encoder_trainable"] != 0
        and not model.preserve_base_asr
    ):
        raise ValueError(
            "CTC loss may be excluded only when the encoder is frozen or the ASR tail is isolated"
        )
    train_data = JointFeatureDataset(
        manifest, split="train", training_target=config.training_target
    )
    validation_data = JointFeatureDataset(
        manifest, split="development", training_target=config.training_target
    )
    generator = torch.Generator().manual_seed(config.seed)
    if config.corpus_balancing:
        sampler = _corpus_balanced_sampler(
            train_data,
            generator=generator,
            num_samples=config.samples_per_epoch,
        )
    else:
        sampler = RandomSampler(
            train_data,
            replacement=config.samples_per_epoch is not None,
            num_samples=config.samples_per_epoch,
            generator=generator,
        )
    train_batches = DurationBucketBatchSampler(
        sampler,
        [row.duration_ms for row in train_data.rows],
        batch_size=config.batch_size,
        bucket_multiplier=config.duration_bucket_multiplier,
    )
    train_loader = DataLoader(
        train_data,
        batch_sampler=train_batches,
        collate_fn=collate_joint_examples,
    )
    validation_batches = DurationBucketBatchSampler(
        SequentialSampler(validation_data),
        [row.duration_ms for row in validation_data.rows],
        batch_size=config.validation_batch_size or config.batch_size,
        bucket_multiplier=max(len(validation_data), 1),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_sampler=validation_batches,
        collate_fn=collate_joint_examples,
    )
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, config), weight_decay=config.weight_decay
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_digest = manifest_sha256(manifest)
    source_digest = _training_source_sha256()
    state_path = output_dir / "trainer-state.pt"
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    stale_epochs = 0
    first_epoch = 1
    accrued_cost = 0.0

    if config.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("manifest_sha256") != manifest_digest:
            raise ValueError("resume state belongs to a different training manifest")
        if state.get("training_source_sha256") != source_digest:
            raise ValueError("training source changed since the resume state was written")
        if state.get("initial_checkpoint_sha256") != initial_checkpoint_sha256:
            raise ValueError("resume state used a different initial checkpoint")
        load_attune_checkpoint(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        generator.set_state(state["generator_state"])
        history = list(state["history"])
        best_validation = float(state["best_validation_loss"])
        stale_epochs = int(state["stale_epochs"])
        first_epoch = int(state["completed_epoch"]) + 1
        accrued_cost = float(state["estimated_cost_gbp"])

    for epoch in range(first_epoch, config.epochs + 1):
        model.train()
        if model.trainable_parameter_summary()["encoder_trainable"] == 0:
            model.sensevoice.eval()
        optimizer.zero_grad(set_to_none=True)
        training_total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            current_cost = accrued_cost + _estimated_cost(started, config.gpu_hour_cost_gbp)
            if current_cost >= config.maximum_cost_gbp:
                raise RuntimeError(
                    "training stopped before exceeding the configured compute budget"
                )
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(
                    batch.speech,
                    batch.speech_lengths,
                    compute_ctc_logits=config.include_ctc_loss,
                )
                targets = _targets_for_loss(batch.targets, include_ctc_loss=config.include_ctc_loss)
                loss, _ = compute_joint_loss(output, targets, weights)
                scaled_loss = loss / config.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            if step % config.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            training_total += float(loss.detach())
            if step % config.log_interval_steps == 0:
                progress = {
                    "phase": "train",
                    "epoch": epoch,
                    "step": step,
                    "steps": len(train_loader),
                    "mean_loss": training_total / step,
                    "estimated_cost_gbp": accrued_cost
                    + _estimated_cost(started, config.gpu_hour_cost_gbp),
                }
                _atomic_json_write(progress, output_dir / "training-progress.json")
                print(json.dumps({"training_progress": progress}), flush=True)

        model.eval()
        validation_total = 0.0
        with torch.inference_mode():
            for batch in validation_loader:
                batch = batch.to(device)
                output = model(
                    batch.speech,
                    batch.speech_lengths,
                    compute_ctc_logits=config.include_ctc_loss,
                )
                targets = _targets_for_loss(batch.targets, include_ctc_loss=config.include_ctc_loss)
                loss, _ = compute_joint_loss(output, targets, weights)
                validation_total += float(loss)
        train_mean = training_total / len(train_loader)
        validation_mean = validation_total / len(validation_loader)
        history.append(
            {"epoch": epoch, "training_loss": train_mean, "validation_loss": validation_mean}
        )
        if validation_mean < best_validation:
            best_validation = validation_mean
            stale_epochs = 0
            _atomic_torch_save(attune_delta_checkpoint(model), output_dir / "model.pt")
        else:
            stale_epochs += 1
        current_cost = accrued_cost + _estimated_cost(started, config.gpu_hour_cost_gbp)
        _atomic_torch_save(
            {
                "format": "attune_trainer_state_v1",
                "manifest_sha256": manifest_digest,
                "training_source_sha256": source_digest,
                "initial_checkpoint_sha256": initial_checkpoint_sha256,
                "completed_epoch": epoch,
                "model": attune_delta_checkpoint(model),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "generator_state": generator.get_state(),
                "history": history,
                "best_validation_loss": best_validation,
                "stale_epochs": stale_epochs,
                "estimated_cost_gbp": current_cost,
            },
            state_path,
        )
        _atomic_json_write(
            {
                "completed_epoch": epoch,
                "best_validation_loss": best_validation,
                "stale_epochs": stale_epochs,
                "estimated_cost_gbp": current_cost,
                "latest": history[-1],
            },
            output_dir / "training-progress.json",
        )
        print(json.dumps({"training_progress": history[-1]}), flush=True)
        if stale_epochs >= config.early_stopping_patience:
            break

    total_cost = accrued_cost + _estimated_cost(started, config.gpu_hour_cost_gbp)
    report = {
        "schema_version": "1.0",
        "manifest_sha256": manifest_digest,
        "training_source_sha256": source_digest,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "trainer": asdict(config),
        "loss_weights": asdict(weights or LossWeights()),
        "parameter_summary": model.trainable_parameter_summary(),
        "preserve_base_asr": model.preserve_base_asr,
        "history": history,
        "best_validation_loss": best_validation,
        "elapsed_seconds": time.monotonic() - started,
        "estimated_cost_gbp": total_cost,
        "checkpoint_format": "attune_delta_v1",
    }
    _atomic_json_write(report, output_dir / "training-report.json")
    return report
