from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path
from types import ModuleType

from attune.baselines.adapters import BaselinePrediction
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


def test_real_mode_runs_sensevoice_only_partial_cascade(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "isolated.wav"
    write_wav(audio)
    sensevoice = tmp_path / "sensevoice-small"
    sensevoice.mkdir()
    json_path = tmp_path / "out.json"
    html_path = tmp_path / "out.html"
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")
    for name in (
        "ATTUNE_SENSEVOICE_SMALL_PATH",
        "ATTUNE_EMOTION2VEC_PLUS_PATH",
        "ATTUNE_VOCALSOUND_PROBE_PATH",
        "ATTUNE_FSD50K_PROBE_PATH",
        "ATTUNE_TEMPORAL_HEAD_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    module = load_script()
    captured: dict[str, object] = {}

    class FakeCascade:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def availability(self) -> tuple[bool, str | None]:
            return True, None

        def predict(self, item):
            prediction = module._fixture_prediction(item.audio_path)
            payload = prediction.output.model_dump(mode="json")
            payload["model"]["name"] = "attune-cascade:sensevoice+affect-abstain+aed"
            return BaselinePrediction(
                output=AttuneOutput.model_validate(payload),
                runtime=prediction.runtime,
                diagnostics=prediction.diagnostics,
            )

    monkeypatch.setattr(module, "AttuneCascade", FakeCascade)
    monkeypatch.setattr(module, "DCASE_CANDIDATES", ())
    result = module.run(
        [
            str(audio),
            "--sensevoice-path",
            str(sensevoice),
            "--output",
            str(json_path),
            "--html-output",
            str(html_path),
        ]
    )

    assert result == 0
    assert captured["emotion2vec_checkpoint"] is None
    assert captured["vocalsound_probe_checkpoint"] is None
    assert captured["fsd50k_probe_checkpoint"] is None
    assert captured["temporal_head_checkpoints"] == ()
    output = AttuneOutput.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert output.affect.abstain is True
    html = html_path.read_text(encoding="utf-8")
    assert "Partial cascade from local artifacts only" in html
    assert "emotion2vec+" in html
    assert "DCASE frame timestamps omitted" in html
    assert "Not a model prediction" not in html


def test_explicit_missing_emotion2vec_path_is_an_error(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    audio = tmp_path / "synthetic.wav"
    write_wav(audio)
    sensevoice = tmp_path / "sensevoice-small"
    sensevoice.mkdir()
    monkeypatch.setenv("ATTUNE_SENSEVOICE_LICENSE_REVIEWED", "1")
    result = load_script().run(
        [
            str(audio),
            "--sensevoice-path",
            str(sensevoice),
            "--emotion2vec-path",
            str(tmp_path / "missing-emotion2vec"),
        ]
    )
    assert result == 2
    assert "explicit optional artifact path is missing" in capsys.readouterr().err


def test_omission_notes_name_dcase_when_configured() -> None:
    module = load_script()
    configured = module._omission_notes(
        ["emotion2vec+", "VocalSound probe", "FSD50K probe"],
        dcase_head_configured=True,
    )
    assert configured
    assert "DCASE frame spans are configured for laugh/cough/throat_clear." in configured[0]
    assert "not DCASE localization" not in configured[0]
    omitted = module._omission_notes(["emotion2vec+"], dcase_head_configured=False)
    assert "This is not DCASE localization." in omitted[0]


def test_explicit_dcase_head_is_passed_to_cascade(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    audio = tmp_path / "isolated.wav"
    write_wav(audio)
    sensevoice = tmp_path / "sensevoice-small"
    sensevoice.mkdir()
    json_path = tmp_path / "out.json"
    html_path = tmp_path / "out.html"
    checkpoint = tmp_path / "frame-head.pt"
    head = torch.nn.Sequential(torch.nn.Linear(512, 128), torch.nn.ReLU(), torch.nn.Linear(128, 3))
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "feature_mean": torch.zeros(512),
            "feature_scale": torch.ones(512),
            "labels": ("laugh", "cough", "throat_clear"),
            "hidden_size": 128,
            "threshold": 0.95,
            "dataset": "dcase2016_task2",
            "embedding": "sensevoice-small-encoder-frames-v1",
            "encoder_frozen": True,
            "gate": {
                "passed": True,
                "margin_required": 0.05,
                "margin_observed": 0.3876,
                "temporal_segment_f1": 0.7059,
                "whole_clip_segment_f1": 0.3183,
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
    module = load_script()
    captured: dict[str, object] = {}

    class FakeCascade:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def availability(self) -> tuple[bool, str | None]:
            return True, None

        def predict(self, item):
            prediction = module._fixture_prediction(item.audio_path)
            payload = prediction.output.model_dump(mode="json")
            payload["model"]["name"] = (
                "attune-cascade:sensevoice+affect-abstain+aed+1-gated-frame-heads"
            )
            payload["events"] = [
                {
                    "id": "e1",
                    "label": "cough",
                    "start_ms": 20,
                    "end_ms": 80,
                    "after_word_id": None,
                    "confidence": 0.9,
                    "status": "provisional",
                }
            ]
            return BaselinePrediction(
                output=AttuneOutput.model_validate(payload),
                runtime=prediction.runtime,
                diagnostics=prediction.diagnostics,
            )

    monkeypatch.setattr(module, "AttuneCascade", FakeCascade)
    result = module.run(
        [
            str(audio),
            "--sensevoice-path",
            str(sensevoice),
            "--temporal-head",
            str(checkpoint),
            "--output",
            str(json_path),
            "--html-output",
            str(html_path),
        ]
    )

    assert result == 0
    assert captured["temporal_head_checkpoints"] == (checkpoint,)
    html = html_path.read_text(encoding="utf-8")
    assert "DCASE frame spans are configured for laugh/cough/throat_clear." in html
    assert "Solid bars are DCASE frame spans" in html
    assert "not DCASE localization" not in html
    assert "STARSS23 stays unwired" not in html or "DCASE frame timestamps omitted" not in html

