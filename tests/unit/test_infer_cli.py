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
