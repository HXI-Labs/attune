from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import SequentialSampler

from attune.models.joint import AdaptationPolicy, AttuneJointModel
from attune.training.losses import JointTargets
from attune.training.trainer import (
    DurationBucketBatchSampler,
    PairedDurationBucketBatchSampler,
    TrainerConfig,
    _corpus_balanced_sampler,
    _restrict_training_target,
    _targets_for_loss,
    train_joint_model,
)


class TinyCTC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ctc_lo = nn.Linear(16, 12)


class TinySenseVoice(nn.Module):
    encoder_output_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.encoders = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
        self.ctc = TinyCTC()
        self.input = nn.Linear(8, 16)

    def encode(self, speech, lengths, rich_tokens):
        del rich_tokens
        acoustic = self.input(speech)
        queries = torch.zeros(speech.shape[0], 4, 16)
        return torch.cat((queries, acoustic), dim=1), lengths + 4


def test_corpus_sampler_assigns_equal_total_mass_to_each_corpus() -> None:
    class FixtureDataset:
        rows = [
            SimpleNamespace(dataset_id="large"),
            SimpleNamespace(dataset_id="large"),
            SimpleNamespace(dataset_id="large"),
            SimpleNamespace(dataset_id="small"),
        ]

        def __len__(self) -> int:
            return len(self.rows)

    sampler = _corpus_balanced_sampler(
        FixtureDataset(), generator=torch.Generator().manual_seed(42)
    )

    assert sampler.weights[:3].sum().item() == 1.0
    assert sampler.weights[3:].sum().item() == 1.0


def test_corpus_sampler_honours_fixed_epoch_size() -> None:
    dataset = SimpleNamespace(
        rows=[
            SimpleNamespace(dataset_id="first"),
            SimpleNamespace(dataset_id="second"),
        ]
    )

    sampler = _corpus_balanced_sampler(
        dataset,
        generator=torch.Generator().manual_seed(42),
        num_samples=7,
    )

    assert len(sampler) == 7


def test_affect_class_balancing_equalizes_classes_within_each_corpus() -> None:
    def row(dataset: str, label: str):
        return SimpleNamespace(
            dataset_id=dataset,
            affect_distribution={
                "neutral": float(label == "neutral"),
                "anger": float(label == "anger"),
            },
        )

    dataset = SimpleNamespace(
        rows=[
            row("first", "neutral"),
            row("first", "neutral"),
            row("first", "neutral"),
            row("first", "anger"),
            row("second", "neutral"),
            row("second", "anger"),
        ]
    )

    sampler = _corpus_balanced_sampler(
        dataset,
        generator=torch.Generator().manual_seed(42),
        num_samples=6,
        affect_class_balancing=True,
    )

    assert sampler.weights[:3].sum().item() == pytest.approx(0.5)
    assert sampler.weights[3].item() == pytest.approx(0.5)
    assert sampler.weights[4].item() == pytest.approx(0.5)
    assert sampler.weights[5].item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "config, message",
    [
        (TrainerConfig(samples_per_epoch=0), "samples_per_epoch"),
        (TrainerConfig(validation_batch_size=0), "validation_batch_size"),
        (TrainerConfig(paired_batch_fraction=1.1), "paired_batch_fraction"),
        (TrainerConfig(affect_class_balancing=True), "affect_class_balancing"),
        (TrainerConfig(training_target="unknown"), "training target"),
    ],
)
def test_trainer_config_rejects_non_positive_optional_batch_limits(
    config: TrainerConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_duration_bucket_sampler_limits_padding_within_batches() -> None:
    durations = [100, 900, 200, 800, 300, 700, 400, 600]
    batches = list(
        DurationBucketBatchSampler(
            SequentialSampler(durations),
            durations,
            batch_size=2,
            bucket_multiplier=4,
        )
    )

    assert len(batches) == 4
    assert all(
        max(durations[index] for index in batch) - min(durations[index] for index in batch) <= 200
        for batch in batches
    )


def test_pair_sampler_constructs_dataset_scoped_contrast_batches() -> None:
    rows = [
        SimpleNamespace(
            dataset_id="paired",
            pair_id=0,
            duration_ms=100,
            affect_distribution={"neutral": 1.0, "anger": 0.0},
        ),
        SimpleNamespace(
            dataset_id="unpaired",
            pair_id=-1,
            duration_ms=110,
            affect_distribution={"neutral": 1.0, "anger": 0.0},
        ),
        SimpleNamespace(
            dataset_id="paired",
            pair_id=0,
            duration_ms=500,
            affect_distribution={"neutral": 0.0, "anger": 1.0},
        ),
        SimpleNamespace(
            dataset_id="unpaired",
            pair_id=-1,
            duration_ms=510,
            affect_distribution={"neutral": 1.0, "anger": 0.0},
        ),
    ]
    sampler = PairedDurationBucketBatchSampler(
        SequentialSampler(rows),
        rows,
        batch_size=2,
        bucket_multiplier=2,
        paired_batch_fraction=1.0,
    )

    batches = list(sampler)

    assert sampler.active_pair_fraction == 1.0
    assert all({0, 2}.issubset(batch) for batch in batches)


def test_pair_sampler_does_not_link_equal_ids_from_different_datasets() -> None:
    rows = [
        SimpleNamespace(
            dataset_id="first",
            pair_id=0,
            duration_ms=100,
            affect_distribution={"neutral": 1.0, "anger": 0.0},
        ),
        SimpleNamespace(
            dataset_id="second",
            pair_id=0,
            duration_ms=110,
            affect_distribution={"neutral": 0.0, "anger": 1.0},
        ),
    ]
    sampler = PairedDurationBucketBatchSampler(
        SequentialSampler(rows),
        rows,
        batch_size=2,
        bucket_multiplier=1,
        paired_batch_fraction=1.0,
    )

    list(sampler)

    assert sampler.active_pair_fraction == 0.0


def test_frozen_training_can_exclude_read_only_ctc_monitor() -> None:
    targets = JointTargets(
        ctc_targets=torch.tensor([1, 2]),
        ctc_target_lengths=torch.tensor([2]),
        ctc_example_mask=torch.tensor([True]),
        affect_distribution=torch.tensor([[1.0, 0.0]]),
    )

    selected = _targets_for_loss(targets, include_ctc_loss=False)

    assert selected.ctc_targets is None
    assert selected.ctc_target_lengths is None
    assert selected.ctc_example_mask is None
    assert selected.affect_distribution is targets.affect_distribution
    assert _targets_for_loss(targets, include_ctc_loss=True) is targets


@pytest.mark.parametrize(
    ("target", "prefixes"),
    [
        ("affect", ("affect_projection.", "affect_head.")),
        ("events", ("event_head.",)),
        ("weak_events", ("event_head.",)),
        ("styles", ("style_projection.", "style_head.")),
    ],
)
def test_targeted_training_freezes_every_unrelated_parameter(
    target: str, prefixes: tuple[str, ...]
) -> None:
    model = AttuneJointModel(TinySenseVoice(), hidden_size=8, affect_embedding_size=4)

    _restrict_training_target(model, target)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith(prefixes) for name in trainable)


def test_targeted_training_retains_configured_encoder_adaptation() -> None:
    model = AttuneJointModel(
        TinySenseVoice(),
        adaptation_policy=AdaptationPolicy.UPPER_TWO,
        hidden_size=8,
        affect_embedding_size=4,
    )

    _restrict_training_target(model, "styles")

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert any(name.startswith("sensevoice.encoder.encoders.") for name in trainable)
    assert all(
        name.startswith(("sensevoice.", "style_projection.", "style_head.")) for name in trainable
    )


def test_trainer_writes_small_delta_checkpoint(tmp_path: Path) -> None:
    feature = tmp_path / "feature.pt"
    torch.save(torch.randn(8, 8), feature)
    digest = hashlib.sha256(feature.read_bytes()).hexdigest()
    categories = {
        name: float(name == "neutral")
        for name in (
            "neutral",
            "joy",
            "distress",
            "anger",
            "fear",
            "surprise",
            "other",
            "ambiguous",
        )
    }
    rows = []
    for split in ("train", "development"):
        rows.append(
            {
                "clip_id": f"{split}-1",
                "dataset_id": "fixture",
                "split": split,
                "speaker_id": f"{split}-speaker",
                "feature_path": feature.name,
                "feature_sha256": digest,
                "duration_ms": 480,
                "frame_hop_ms": 60,
                "events": [],
                "event_presence": [],
                "styles": [],
                "affect_distribution": categories,
                "vad": [0.0, 0.0, 0.0],
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    model = AttuneJointModel(TinySenseVoice(), hidden_size=8, affect_embedding_size=4)
    output = tmp_path / "run"

    report = train_joint_model(
        model,
        manifest=manifest,
        output_dir=output,
        config=TrainerConfig(
            epochs=1,
            batch_size=1,
            device="cpu",
            gpu_hour_cost_gbp=0.0,
        ),
    )
    checkpoint = torch.load(output / "model.pt", weights_only=True)

    assert report["checkpoint_format"] == "attune_delta_v1"
    assert checkpoint["format"] == "attune_delta_v1"
    assert checkpoint["state_dict"]
    assert all(not name.startswith("sensevoice.") for name in checkpoint["state_dict"])

    resumed_model = AttuneJointModel(TinySenseVoice(), hidden_size=8, affect_embedding_size=4)
    resumed = train_joint_model(
        resumed_model,
        manifest=manifest,
        output_dir=output,
        config=TrainerConfig(
            epochs=2,
            batch_size=1,
            device="cpu",
            gpu_hour_cost_gbp=0.0,
        ),
    )

    assert [item["epoch"] for item in resumed["history"]] == [1, 2]
    assert (output / "training-progress.json").is_file()
    state = torch.load(output / "trainer-state.pt", weights_only=True)
    assert state["format"] == "attune_trainer_state_v1"
    assert state["completed_epoch"] == 2
