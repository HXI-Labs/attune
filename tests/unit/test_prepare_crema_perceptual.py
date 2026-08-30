from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path("scripts/prepare_crema_perceptual.py")
    spec = importlib.util.spec_from_file_location("prepare_crema_perceptual", path)
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(path.parent.resolve()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_responses(path: Path, ratings: list[dict[str, str]]) -> None:
    fields = ("queryType", "clipName", "respEmo")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ratings)


def test_crema_perceptual_rows_use_voice_rater_distributions(tmp_path: Path) -> None:
    module = _module()
    responses = tmp_path / "responses.csv"
    _write_responses(
        responses,
        [
            {"queryType": "1", "clipName": "1031_DFA_ANG_XX", "respEmo": "A"},
            {"queryType": "1", "clipName": "1031_DFA_ANG_XX", "respEmo": "A"},
            {"queryType": "1", "clipName": "1031_DFA_ANG_XX", "respEmo": "N"},
            {"queryType": "2", "clipName": "1031_DFA_ANG_XX", "respEmo": "H"},
            {"queryType": "1", "clipName": "1081_DFA_ANG_XX", "respEmo": "A"},
        ],
    )

    rows = module.protocol_rows(responses)

    assert len(rows) == 1
    assert rows[0]["partition"] == "train"
    assert rows[0]["voice_rating_count"] == 3
    assert rows[0]["affect_distribution"]["anger"] == 2 / 3
    assert rows[0]["affect_distribution"]["neutral"] == 1 / 3
    assert sum(rows[0]["affect_distribution"].values()) == 1


def test_crema_perceptual_partitions_preserve_existing_actor_roles(tmp_path: Path) -> None:
    module = _module()
    responses = tmp_path / "responses.csv"
    _write_responses(
        responses,
        [
            {"queryType": "1", "clipName": "1051_IEO_HAP_HI", "respEmo": "H"},
            {"queryType": "1", "clipName": "1061_IEO_HAP_HI", "respEmo": "H"},
            {"queryType": "1", "clipName": "1071_IEO_HAP_HI", "respEmo": "H"},
        ],
    )

    rows = module.protocol_rows(responses)

    assert [row["partition"] for row in rows] == ["train", "development", "train"]
