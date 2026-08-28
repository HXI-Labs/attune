import importlib.util
from pathlib import Path
from types import ModuleType


def load_lastlayer_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "train_dcase_encoder_lastlayer.py"
    spec = importlib.util.spec_from_file_location("dcase_encoder_lastlayer_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_and_decoder_are_locked_to_wired_dcase_head() -> None:
    script = load_lastlayer_script()
    assert script.GATE_EXACT_COLLAR_F1 == 0.4637
    assert script.GATE_HYSTERESIS_COLLAR_F1 == 0.5279
    assert script.CLEAR_SEGMENT_F1_MARGIN == 0.05
    assert script.LOCKED_EXACT_THRESHOLD == 0.95
    assert script.LOCKED_HYSTERESIS_DECODER == {
        "high_threshold": 0.95,
        "low_threshold": 0.855,
        "max_gap_frames": 1,
        "min_active_frames": 1,
    }
    assert (
        script.WIRED_HEAD_SHA256
        == "cb74b1d493511d384dea3abf37139cf754a16f540b594b7f92f82647204b01fc"
    )


def test_replace_rule_does_not_lower_the_gate() -> None:
    script = load_lastlayer_script()
    assert script.should_replace_wired_head(0.4637, 0.5279, 0.05) is True
    assert script.should_replace_wired_head(0.4636, 0.60, 0.40) is False
    assert script.should_replace_wired_head(0.50, 0.5278, 0.40) is False
    assert script.should_replace_wired_head(0.50, 0.60, 0.049) is False


def test_protocol_was_written_before_train_and_excludes_starss23() -> None:
    script = load_lastlayer_script()
    protocol = Path(__file__).parents[2] / "research" / "dcase-encoder-lastlayer.md"
    text = protocol.read_text(encoding="utf-8")
    script.require_protocol(protocol)
    assert "100 ms" in text
    assert "STARSS23 stays unwired" in text
    assert "Do not grid-search" in text
    assert "select_hysteresis_decoder" not in Path(__file__).parents[2].joinpath(
        "scripts/train_dcase_encoder_lastlayer.py"
    ).read_text(encoding="utf-8")
