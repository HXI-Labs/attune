from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path
from types import ModuleType

from attune.schema.output import AttuneOutput


def load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "infer.py"
    spec = importlib.util.spec_from_file_location("infer_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 1_600)


def test_fixture_mode_runs_cli_without_weights(
    tmp_path: Path,
    capsys,
) -> None:
    audio = tmp_path / "synthetic.wav"
    write_wav(audio)

    result = load_script().run([str(audio), "--fixture-mode"])

    captured = capsys.readouterr()
    assert result == 0
    output = AttuneOutput.model_validate(json.loads(captured.out))
    assert output.model.name == "attune-cli-fixture-placeholder"
    assert output.affect.abstain is True
    assert output.affect.top_label is None


def test_real_mode_refuses_unreviewed_sensevoice(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    audio = tmp_path / "synthetic.wav"
    write_wav(audio)
    monkeypatch.delenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", raising=False)

    result = load_script().run([str(audio)])

    assert result == 2
    assert "ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1" in capsys.readouterr().err


def test_xml_stdout_requires_json_output(
    tmp_path: Path,
    capsys,
) -> None:
    audio = tmp_path / "synthetic.wav"
    write_wav(audio)

    result = load_script().run([str(audio), "--fixture-mode", "--xml-output", "-"])

    assert result == 2
    assert "authoritative JSON" in capsys.readouterr().err


def test_fixture_mode_writes_json_xml_and_html(tmp_path: Path) -> None:
    audio = tmp_path / "synthetic.wav"
    write_wav(audio)
    json_path = tmp_path / "out.json"
    xml_path = tmp_path / "out.xml"
    html_path = tmp_path / "out.html"

    result = load_script().run(
        [
            str(audio),
            "--fixture-mode",
            "--output",
            str(json_path),
            "--xml-output",
            str(xml_path),
            "--html-output",
            str(html_path),
        ]
    )

    assert result == 0
    output = AttuneOutput.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert output.model.name == "attune-cli-fixture-placeholder"
    xml = xml_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "<attune" in xml
    assert "Not a model prediction" in html
    assert "DCASE frame timestamps omitted" in html
    assert "does not interpolate words" in html
    assert "synthetic.wav" in html


def test_html_stdout_requires_json_output(tmp_path: Path, capsys) -> None:
    audio = tmp_path / "synthetic.wav"
    write_wav(audio)

    result = load_script().run([str(audio), "--fixture-mode", "--html-output", "-"])

    assert result == 2
    assert "authoritative JSON" in capsys.readouterr().err


def test_explicit_starss23_temporal_head_is_refused(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    audio = tmp_path / "synthetic.wav"
    write_wav(audio)
    checkpoint = tmp_path / "starss23-head.pt"
    head = torch.nn.Sequential(torch.nn.Linear(512, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "feature_mean": torch.zeros(512),
            "feature_scale": torch.ones(512),
            "labels": ("laugh",),
            "hidden_size": 64,
            "threshold": 0.5,
            "dataset": "starss23",
            "embedding": "sensevoice-small-encoder-frames-v1",
            "encoder_frozen": True,
            "gate": {
                "passed": True,
                "segment_margin_required": 0.05,
                "segment_margin_observed": 0.25,
                "collar_f1_required": 0.25,
                "collar_f1_observed": 0.26,
            },
        },
        checkpoint,
    )
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")
    for name in (
        "ATTUNE_SENSEVOICE_SMALL_PATH",
        "ATTUNE_EMOTION2VEC_PLUS_PATH",
        "ATTUNE_VOCALSOUND_PROBE_PATH",
        "ATTUNE_FSD50K_PROBE_PATH",
        "ATTUNE_TEMPORAL_HEAD_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    result = load_script().run([str(audio), "--temporal-head", str(checkpoint)])

    assert result == 2
    assert "STARSS23 timestamps stay unwired" in capsys.readouterr().err
