from pathlib import Path
from types import SimpleNamespace

import pytest

from attune.models.probe_inference import (
    FSD50K_LABEL_MAPPING,
    FrozenLinearProbeHead,
)
from attune.models.sensevoice_probe import SENSEVOICE_EMBEDDING, FrozenSenseVoiceEncoder

torch = pytest.importorskip("torch")


def test_frozen_linear_probe_maps_scream_without_shouting(tmp_path: Path) -> None:
    labels = ["shout", "whisper", "sob", "scream"]
    feature_count = FrozenSenseVoiceEncoder.output_size
    linear = torch.nn.Linear(feature_count, len(labels))
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 2.0]))
    checkpoint = tmp_path / "head.pt"
    torch.save(
        {
            "head_state_dict": linear.state_dict(),
            "feature_mean": torch.zeros(feature_count),
            "feature_scale": torch.ones(feature_count),
            "labels": labels,
            "embedding": SENSEVOICE_EMBEDDING,
        },
        checkpoint,
    )
    provider = SimpleNamespace(
        availability=lambda: (True, None),
        get=lambda: (lambda _path: torch.zeros(feature_count), torch),
    )
    head = FrozenLinearProbeHead(
        name="fsd50k-test",
        checkpoint=checkpoint,
        encoder=provider,
        label_mapping=FSD50K_LABEL_MAPPING,
    )

    prediction = head.predict(tmp_path / "unused.wav")

    assert prediction.annotations[0].label == "scream"
    assert prediction.annotations[0].channel == "event"
    assert prediction.diagnostics["mapped_label"] == "scream"
    assert prediction.diagnostics["confidence_role"].startswith("diagnostic")
