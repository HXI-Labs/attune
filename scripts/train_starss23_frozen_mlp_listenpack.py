#!/usr/bin/env python3
"""Retrain the frozen first-60s STARSS23 laugh MLP on 100 ms overlays.

Copies the 27df8e1 40-epoch frozen MLP + the locked 0a27733 decoder that
produced reported best collar 0.1395. Inspection is the gold-review pack.
Decoder is not searched. Encoder stays frozen. STARSS23 stays unwired unless
collar F1 >= 0.25 and segment margin >= 0.05.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from attune.data.starss23_gold_pack import (
    EXPECTED_FIRST_60S_CLIPS,
    EXPECTED_FIRST_60S_LAUGH_EVENTS,
    EXPECTED_FIRST_60S_TRUE_NEGATIVES,
    HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD,
    assert_expected_pack_counts,
    first_60s_rows,
    is_true_negative,
    laugh_event_count,
    laugh_events,
    resolve_pack_audio,
    sha256_file,
    wav_peak_envelope,
)
from attune.evaluation.localization import (
    collar_event_metrics,
    segment_f1,
    whole_clip_predictions,
)
from attune.models.sensevoice_probe import (
    SENSEVOICE_FRAME_EMBEDDING,
    FrozenSenseVoiceFrameEncoder,
)

LABELS = ("laugh",)
DURATION_MS = 60_000
VALIDATION_ROOMS = {"sony-room21", "tau-room6"}
SEGMENT_MARGIN_REQUIRED = 0.05
COLLAR_F1_REQUIRED = 0.25
PROTOCOL = "first_60s_scene_raster"
SPLIT_PROTOCOL = "development_first60s_val_rooms_sony21_tau6"
MLP_HIDDEN_SIZE = 64
FIRST_60S_TRAIN_CLIPS = 43
FIRST_60S_TRAIN_EVENTS = 51
FIRST_60S_VAL_CLIPS = 19
FIRST_60S_VAL_EVENTS = 14
PRIOR_BEST_COLLAR_F1 = 0.13953488372093023
PRIOR_BEST_PASS = {
    "id": "decoder_validity_0a27733",
    "epochs_completed": 40,
    "segment_f1": 0.512396694214876,
    "whole_clip_segment_f1": 0.17212490479817213,
    "collar_event_f1": PRIOR_BEST_COLLAR_F1,
    "collar_true_positive": 9,
    "collar_false_positive": 72,
    "collar_false_negative": 39,
    "reference_event_count": 48,
    "checkpoint": "artifacts/starss23-scene-raster/frame-head-40epoch.pt",
}
PREDECLARED_DECODER = {
    "high_threshold": 0.95,
    "low_threshold": 0.855,
    "max_gap_frames": 0,
    "min_active_frames": 1,
    "median_filter_frames": 3,
    "onset_shift_ms": 0,
}
DECODER_LOCK_NOTE = (
    "Decoder locked to 0a27733 (high=0.95, low=0.855, gap=0, min_active=1, "
    "median=3, shift=0). Copied from the JSON that produced 0.1395; not searched."
)
FROZEN_40EPOCH_CHECKPOINT = Path("artifacts/starss23-scene-raster/frame-head-40epoch.pt")
NEW_CHECKPOINT = Path("artifacts/starss23-frozen-mlp-listenpack/frame-head.pt")
LISTENPACK_DIR = Path("research/starss23-frozen-mlp-listenpack")
LISTENPACK_CAPTION = (
    "STARSS23 is unwired in product infer; this pack is a research listen test. "
    "Orange = STARSS23 100 ms overlay (owner-authorized activity label, not "
    "second-listener gold). Cyan = frozen-MLP predicted spans. Local paths only."
)
PREFERRED_LAUGH_CLIP_IDS = (
    "starss23-scene-inspection_test-fold4_room23_mix005-w000000",
    "starss23-scene-inspection_test-fold4_room23_mix006-w000000",
    "starss23-scene-inspection_test-fold4_room23_mix007-w000000",
)
PREFERRED_TRUE_NEGATIVE_CLIP_IDS = (
    "starss23-scene-inspection_test-fold4_room23_mix010-w000000",
    "starss23-scene-inspection_test-fold4_room23_mix011-w000000",
)
LISTENPACK_LAUGH_CLIPS = 6
LISTENPACK_TRUE_NEGATIVE_CLIPS = 4


def load_scene_raster_module() -> Any:
    """Load shared hysteresis / MLP helpers from the merged first-60s/tiled trainer."""
    path = Path(__file__).resolve().parent / "train_starss23_scene_raster.py"
    spec = importlib.util.spec_from_file_location("train_starss23_scene_raster", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def should_wire_starss23_timestamps(
    *,
    segment_f1: float,
    whole_clip_segment_f1: float,
    collar_f1: float,
) -> bool:
    """Require collar F1 and a clear segment margin; do not lower the bar after seeing scores."""
    margin = segment_f1 - whole_clip_segment_f1
    return margin >= SEGMENT_MARGIN_REQUIRED and collar_f1 >= COLLAR_F1_REQUIRED


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def first_60s_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["source_window_start_ms"]) == 0]


def event_count(rows: list[dict[str, Any]]) -> int:
    return sum(len(row["events"]) for row in rows)


def index_wavs_by_sha256(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for wav in sorted(root.rglob("*.wav")):
            if not wav.is_file():
                continue
            digest_value = sha256_file(wav)
            index.setdefault(digest_value, wav)
    return index


def resolve_row_audio(row: dict[str, Any], cache_roots: list[Path], hash_index: dict[str, Path]) -> Path:
    expected = row["sha256"]
    relative = Path(row["cache_path"])
    for cache_dir in cache_roots:
        for candidate in (cache_dir / relative, cache_dir / relative.name):
            if candidate.is_file():
                if sha256_file(candidate) != expected:
                    raise RuntimeError(f"sha256 mismatch for {candidate}")
                return candidate
    indexed = hash_index.get(expected)
    if indexed is None:
        raise RuntimeError(f"missing STARSS23 clip for {row['clip_id']}")
    return indexed


def load_rows(manifest: Path, cache_roots: list[Path], hash_index: dict[str, Path]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if set(event["label"] for event in row["events"]) - set(LABELS):
            raise RuntimeError("STARSS23 manifest contains an unsupported Attune mapping")
        row["_audio"] = resolve_row_audio(row, cache_roots, hash_index)
    return rows


def assert_no_later_tiles_in_fit(rows: list[dict[str, Any]], *, split: str) -> None:
    later = [row["clip_id"] for row in rows if int(row["source_window_start_ms"]) != 0]
    if later:
        raise RuntimeError(f"STARSS23 {split} includes later tiles: {later[:3]}")


def assert_inspection_held_out(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    inspection: list[dict[str, Any]],
) -> None:
    fit = train_rows + validation_rows
    fit_ids = {row["clip_id"] for row in fit}
    fit_files = {row["source_recording"] for row in fit}
    fit_rooms = {row["room"] for row in fit}
    insp_ids = {row["clip_id"] for row in inspection}
    insp_files = {row["source_recording"] for row in inspection}
    insp_rooms = {row["room"] for row in inspection}
    if fit_ids & insp_ids:
        raise RuntimeError("STARSS23 inspection clips overlap train/val")
    if fit_files & insp_files:
        raise RuntimeError("STARSS23 inspection files overlap development")
    if fit_rooms & insp_rooms:
        raise RuntimeError("STARSS23 inspection rooms overlap development")


def assert_first_60s_counts(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    inspection: list[dict[str, Any]],
) -> None:
    if len(train_rows) != FIRST_60S_TRAIN_CLIPS or event_count(train_rows) != FIRST_60S_TRAIN_EVENTS:
        raise RuntimeError(
            f"train first-60s drifted: {len(train_rows)} clips / {event_count(train_rows)} events"
        )
    if len(validation_rows) != FIRST_60S_VAL_CLIPS or event_count(validation_rows) != FIRST_60S_VAL_EVENTS:
        raise RuntimeError(
            "val first-60s drifted: "
            f"{len(validation_rows)} clips / {event_count(validation_rows)} events"
        )
    assert_expected_pack_counts(inspection)
    if event_count(inspection) != EXPECTED_FIRST_60S_LAUGH_EVENTS:
        raise RuntimeError("inspection event count drifted from the locked 48-event pack")
    negatives = sum(1 for row in inspection if is_true_negative(row))
    if negatives != EXPECTED_FIRST_60S_TRUE_NEGATIVES:
        raise RuntimeError("inspection true-negative count drifted")


def assert_encoder_frozen(encoder: Any) -> None:
    trainable = sum(parameter.numel() for parameter in encoder.model.parameters() if parameter.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"SenseVoice encoder is not frozen; trainable={trainable}")
    if int(encoder.metadata()["trainable_parameters"]) != 0:
        raise RuntimeError("SenseVoice metadata does not attest 0 trainable parameters")


def assert_checkpoint_does_not_overwrite_40epoch(checkpoint_output: Path) -> None:
    if checkpoint_output.resolve() == FROZEN_40EPOCH_CHECKPOINT.resolve():
        raise RuntimeError("must not overwrite frame-head-40epoch.pt")
    if checkpoint_output.name == "frame-head-40epoch.pt":
        raise RuntimeError("must not write a file named frame-head-40epoch.pt")


def split_development(development: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first60 = first_60s_subset(development)
    train_rows = [row for row in first60 if row["room"] not in VALIDATION_ROOMS]
    validation_rows = [row for row in first60 if row["room"] in VALIDATION_ROOMS]
    return train_rows, validation_rows


def select_listenpack_clips(
    inspection: list[dict[str, Any]],
    *,
    laugh_n: int = LISTENPACK_LAUGH_CLIPS,
    true_negative_n: int = LISTENPACK_TRUE_NEGATIVE_CLIPS,
) -> list[dict[str, Any]]:
    """Pick 8–12 inspection clips: preferred Jerry first-pass, then other rooms."""
    laughs = [row for row in inspection if laugh_event_count([row]) > 0]
    negatives = [row for row in inspection if is_true_negative(row)]
    laughs_by_id = {row["clip_id"]: row for row in laughs}
    negatives_by_id = {row["clip_id"]: row for row in negatives}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(row: dict[str, Any]) -> None:
        if row["clip_id"] in seen:
            return
        selected.append(row)
        seen.add(row["clip_id"])

    for clip_id in PREFERRED_LAUGH_CLIP_IDS:
        if clip_id in laughs_by_id:
            take(laughs_by_id[clip_id])
    remaining_laughs = sorted(
        (row for row in laughs if row["clip_id"] not in seen),
        key=lambda row: (row["room"], row["clip_id"]),
    )
    for row in remaining_laughs:
        if sum(laugh_event_count([item]) > 0 for item in selected) >= laugh_n:
            break
        take(row)
    for clip_id in PREFERRED_TRUE_NEGATIVE_CLIP_IDS:
        if clip_id in negatives_by_id:
            take(negatives_by_id[clip_id])
    remaining_negatives = sorted(
        (row for row in negatives if row["clip_id"] not in seen),
        key=lambda row: (row["room"], row["clip_id"]),
    )
    for row in remaining_negatives:
        if sum(is_true_negative(item) for item in selected) >= true_negative_n:
            break
        take(row)
    total = len(selected)
    if total < 8 or total > 12:
        raise RuntimeError(f"listen pack must have 8–12 clips, got {total}")
    laugh_count = sum(laugh_event_count([row]) > 0 for row in selected)
    negative_count = sum(is_true_negative(row) for row in selected)
    if laugh_count < 1 or negative_count < 1:
        raise RuntimeError("listen pack must mix true laughs and true negatives")
    return selected


def _style() -> str:
    return """
:root { color-scheme: dark; --bg:#10141c; --panel:#1a2130; --ink:#e8eef8; --muted:#9aa8bc;
        --line:#2a3548; --accent:#7db4ff; --warn:#f3c16b; --pred:#5eead4; }
* { box-sizing: border-box; }
body { margin:0; font: 15px/1.45 ui-sans-serif, system-ui, sans-serif; background:var(--bg);
       color:var(--ink); }
a { color:var(--accent); }
header, main { max-width: 1100px; margin: 0 auto; padding: 1.2rem; }
.banner { background:#3a2a12; color:var(--warn); padding:.75rem 1rem; border:1px solid #6b5424;
          border-radius:8px; margin-bottom:1rem; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:1rem;
         margin:1rem 0; }
.muted { color:var(--muted); }
.legend span { display:inline-block; width:12px; height:12px; margin-right:.35rem;
               vertical-align:middle; }
.orange { background:rgba(243,193,107,0.85); }
.cyan { background:rgba(94,234,212,0.85); }
canvas { width:100%; height:140px; background:#0b1018; border-radius:8px; }
audio { width:100%; margin:.6rem 0; }
table { width:100%; border-collapse:collapse; }
th, td { text-align:left; padding:.4rem .5rem; border-bottom:1px solid var(--line); }
""".strip()


def _app_js() -> str:
    return r"""
function drawWave(canvas, peaks, durationMs, overlays, predictions, playheadMs) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(1, canvas.clientWidth);
  const cssH = Math.max(1, canvas.clientHeight);
  const needW = Math.round(cssW * dpr);
  const needH = Math.round(cssH * dpr);
  if (canvas.width !== needW || canvas.height !== needH) {
    canvas.width = needW; canvas.height = needH;
  }
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#7db4ff";
  const mid = h / 2;
  peaks.forEach((peak, i) => {
    const x = i / peaks.length * w;
    const mag = peak * (h * 0.42);
    ctx.fillRect(x, mid - mag, Math.max(1, w / peaks.length), mag * 2);
  });
  function mark(start, end, color) {
    const x0 = start / durationMs * w;
    const x1 = end / durationMs * w;
    ctx.fillStyle = color;
    ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
  }
  overlays.forEach(ev => mark(ev.start_ms, ev.end_ms, "rgba(243,193,107,0.32)"));
  predictions.forEach(ev => mark(ev.start_ms, ev.end_ms, "rgba(94,234,212,0.28)"));
  if (playheadMs != null && durationMs > 0) {
    const x = (playheadMs / durationMs) * w;
    ctx.fillStyle = "#cf222e";
    ctx.fillRect(Math.max(0, x - dpr), 0, Math.max(2 * dpr, dpr), h);
  }
}
function boot(clip) {
  const canvas = document.getElementById("wave");
  const audio = document.querySelector("audio");
  let playheadMs = 0;
  function redraw() {
    if (canvas && clip.peaks) {
      drawWave(canvas, clip.peaks, clip.duration_ms, clip.overlays_100ms,
               clip.predicted_spans, playheadMs);
    }
  }
  redraw();
  window.addEventListener("resize", redraw);
  if (audio) {
    audio.addEventListener("timeupdate", () => {
      playheadMs = Math.round(audio.currentTime * 1000);
      const now = document.getElementById("now_ms");
      if (now) now.textContent = String(playheadMs);
      redraw();
    });
  }
}
"""


def render_clip_html(payload: dict[str, Any]) -> str:
    audio = (
        f"<audio controls src='{escape(payload['audio_href'])}'></audio>"
        if payload.get("audio_href")
        else "<p class='banner'>Audio missing locally; waveform peaks still shown.</p>"
    )
    overlay_rows = "".join(
        f"<tr><td>100 ms overlay</td><td>{ev['start_ms']}–{ev['end_ms']}</td><td>{escape(ev['label'])}</td></tr>"
        for ev in payload["overlays_100ms"]
    ) or "<tr><td colspan='3'>True negative — no 100 ms laugh overlay</td></tr>"
    pred_rows = "".join(
        f"<tr><td>predicted</td><td>{ev['start_ms']}–{ev['end_ms']}</td><td>{escape(ev['label'])}</td></tr>"
        for ev in payload["predicted_spans"]
    ) or "<tr><td colspan='3'>No predicted laugh spans</td></tr>"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{escape(payload["clip_id"])}</title>
<style>{_style()}</style></head>
<body>
<header>
  <p><a href="index.html">← listen pack</a></p>
  <div class="banner">{escape(LISTENPACK_CAPTION)}</div>
  <h1>{escape(payload["clip_id"])}</h1>
  <p class="muted">{escape(str(payload.get("source_recording")))} ·
  {escape(str(payload.get("room")))} · {payload["duration_ms"]} ms ·
  true_negative={payload["true_negative"]}</p>
</header>
<main>
  <div class="panel">
    {audio}
    <canvas id="wave"></canvas>
    <p><strong>Now <span id="now_ms">0</span> ms</strong></p>
    <p class="legend"><span class="orange"></span> orange 100 ms overlay
       &nbsp; <span class="cyan"></span> model predicted spans</p>
  </div>
  <div class="panel">
    <table>
      <thead><tr><th>kind</th><th>span ms</th><th>label</th></tr></thead>
      <tbody>{overlay_rows}{pred_rows}</tbody>
    </table>
  </div>
</main>
<script>{_app_js()}</script>
<script>boot({json.dumps(payload, ensure_ascii=True)});</script>
</body></html>
"""


def render_index_html(clips: list[dict[str, Any]], *, wired: bool) -> str:
    rows = []
    for clip in clips:
        n_over = len(clip["overlays_100ms"])
        n_pred = len(clip["predicted_spans"])
        kind = "true negative" if clip["true_negative"] else f"{n_over} overlay laugh(s)"
        rows.append(
            "<tr>"
            f"<td><a href='{escape(clip['html_name'])}'>{escape(clip['clip_id'])}</a></td>"
            f"<td>{escape(str(clip.get('room')))}</td>"
            f"<td>{kind}</td>"
            f"<td>{n_pred} predicted</td>"
            "</tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>STARSS23 frozen MLP listen pack</title>
<style>{_style()}</style></head>
<body>
<header>
  <div class="banner">{escape(LISTENPACK_CAPTION)} wired={wired}.</div>
  <h1>STARSS23 frozen first-60s MLP listen test</h1>
  <p class="muted">Local files only. Do not public-tunnel. Wavs are gitignored.</p>
</header>
<main>
  <div class="panel">
    <table>
      <thead><tr><th>clip</th><th>room</th><th>100 ms overlays</th><th>model</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</main>
</body></html>
"""


def write_listen_pack(
    selected: list[dict[str, Any]],
    predictions_by_id: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    *,
    copy_audio: bool,
    wired: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    written = []
    copied = 0
    for row in selected:
        clip_id = row["clip_id"]
        source = Path(row["_audio"])
        peaks = wav_peak_envelope(source)
        audio_href = None
        if copy_audio:
            destination = audio_dir / f"{clip_id}.wav"
            shutil.copyfile(source, destination)
            copied += 1
            audio_href = f"audio/{clip_id}.wav"
        overlays = [
            {
                "start_ms": int(event["start_ms"]),
                "end_ms": int(event["end_ms"]),
                "label": "laugh",
                "source": "starss23_100ms",
            }
            for event in laugh_events(row)
        ]
        predicted = predictions_by_id.get(clip_id, [])
        html_name = f"{clip_id}.html"
        payload = {
            "schema_version": "1.0",
            "kind": "starss23-frozen-mlp-listen-test",
            "wired": wired,
            "caption": LISTENPACK_CAPTION,
            "clip_id": clip_id,
            "html_name": html_name,
            "source_recording": row.get("source_recording"),
            "room": row.get("room"),
            "duration_ms": int(row["duration_ms"]),
            "sha256": row["sha256"],
            "true_negative": is_true_negative(row),
            "label_status": HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD,
            "audio_href": audio_href,
            "peaks": peaks,
            "overlays_100ms": overlays,
            "predicted_spans": predicted,
            "gold": False,
        }
        (output_dir / html_name).write_text(render_clip_html(payload), encoding="utf-8")
        (output_dir / f"{clip_id}.json").write_text(
            json.dumps({k: v for k, v in payload.items() if k != "peaks"}, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(payload)
    (output_dir / "index.html").write_text(render_index_html(written, wired=wired), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "kind": "starss23-frozen-mlp-listen-test",
        "caption": LISTENPACK_CAPTION,
        "wired": wired,
        "n_clips": len(written),
        "n_true_laugh_clips": sum(not clip["true_negative"] for clip in written),
        "n_true_negative_clips": sum(clip["true_negative"] for clip in written),
        "copied_wavs": copied,
        "clips": [
            {
                "clip_id": clip["clip_id"],
                "html": clip["html_name"],
                "json": f"{clip['clip_id']}.json",
                "true_negative": clip["true_negative"],
                "n_overlays_100ms": len(clip["overlays_100ms"]),
                "n_predicted_spans": len(clip["predicted_spans"]),
            }
            for clip in written
        ],
        "local_paths_only": True,
        "public_tunnel": False,
    }
    (output_dir / "listenpack.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_fixture_htmls(output_dir: Path) -> None:
    """Two small committed fixtures so the pack is reviewable without wavs."""
    fixtures = output_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    laugh = {
        "schema_version": "1.0",
        "kind": "starss23-frozen-mlp-listen-test",
        "wired": False,
        "caption": LISTENPACK_CAPTION,
        "clip_id": "fixture-true-laugh",
        "html_name": "true-laugh.attune.html",
        "source_recording": "fixture.wav",
        "room": "fixture-room",
        "duration_ms": 4000,
        "sha256": "0" * 64,
        "true_negative": False,
        "label_status": HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD,
        "audio_href": None,
        "peaks": [0.1, 0.4, 0.8, 0.3, 0.2, 0.1, 0.05, 0.02],
        "overlays_100ms": [{"start_ms": 800, "end_ms": 1600, "label": "laugh", "source": "starss23_100ms"}],
        "predicted_spans": [{"start_ms": 960, "end_ms": 1500, "label": "laugh"}],
        "gold": False,
    }
    negative = {
        **laugh,
        "clip_id": "fixture-true-negative",
        "html_name": "true-negative.attune.html",
        "true_negative": True,
        "overlays_100ms": [],
        "predicted_spans": [],
        "peaks": [0.05, 0.08, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01],
    }
    (fixtures / "true-laugh.attune.html").write_text(render_clip_html(laugh), encoding="utf-8")
    (fixtures / "true-laugh.attune.json").write_text(
        json.dumps({k: v for k, v in laugh.items() if k != "peaks"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (fixtures / "true-negative.attune.html").write_text(render_clip_html(negative), encoding="utf-8")
    (fixtures / "true-negative.attune.json").write_text(
        json.dumps({k: v for k, v in negative.items() if k != "peaks"}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("data/manifests/starss23-scene-raster-development.jsonl"),
    )
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/starss23-gold-review-pack.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        action="append",
        dest="cache_dirs",
        default=None,
        help="Audio cache root; repeatable. Defaults to the first-60s mean-of-4 cache.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/starss23-scene-raster/embeddings"),
    )
    parser.add_argument("--checkpoint-output", type=Path, default=NEW_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/starss23-frozen-mlp-listenpack-results.json"),
    )
    parser.add_argument("--listenpack-dir", type=Path, default=LISTENPACK_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-encode", action="store_true")
    parser.add_argument("--copy-listenpack-audio", action="store_true", default=True)
    parser.add_argument("--no-copy-listenpack-audio", action="store_false", dest="copy_listenpack_audio")
    arguments = parser.parse_args()
    import torch

    os.environ["ATTUNE_SENSEVOICE_LICENSE_REVIEWED"] = "1"
    if torch.cuda.is_available():
        raise RuntimeError("this pass is CPU only")
    torch.manual_seed(arguments.seed)
    raster = load_scene_raster_module()
    if raster.PREDECLARED_DECODER != PREDECLARED_DECODER:
        raise RuntimeError("locked decoder drifted from the 0.1395 0a27733 cell")
    if raster.COLLAR_F1_REQUIRED != COLLAR_F1_REQUIRED or raster.SEGMENT_MARGIN_REQUIRED != SEGMENT_MARGIN_REQUIRED:
        raise RuntimeError("wiring gates must stay 0.25 / 0.05")
    assert_checkpoint_does_not_overwrite_40epoch(arguments.checkpoint_output)

    cache_dirs = list(arguments.cache_dirs or [Path("data/raw/starss23-scene-raster")])
    extra = Path("data/raw/starss23-scene-raster-tiled")
    if extra.exists() and extra not in cache_dirs:
        cache_dirs.append(extra)
    hash_index = index_wavs_by_sha256(cache_dirs)
    development = first_60s_subset(load_rows(arguments.development_manifest, cache_dirs, hash_index))
    inspection = first_60s_rows(load_rows(arguments.inspection_manifest, cache_dirs, hash_index))
    train_rows, validation_rows = split_development(development)
    if not train_rows or not validation_rows:
        raise RuntimeError("STARSS23 scene split produced an empty partition")
    if {row["room"] for row in train_rows} & {row["room"] for row in validation_rows}:
        raise RuntimeError("STARSS23 validation rooms overlap training")
    assert_no_later_tiles_in_fit(train_rows, split="train")
    assert_no_later_tiles_in_fit(validation_rows, split="val")
    assert_inspection_held_out(train_rows, validation_rows, inspection)
    assert_first_60s_counts(train_rows, validation_rows, inspection)
    raster.assert_whole_files_stay_in_one_split(train_rows, validation_rows)

    encoder = FrozenSenseVoiceFrameEncoder(
        arguments.sensevoice_path,
        arguments.embedding_cache,
        torch,
    )
    assert_encoder_frozen(encoder)

    def features(rows: list[dict[str, Any]]) -> list[Any]:
        return [encoder(row["_audio"]) for row in rows]

    train_x = features(train_rows)
    validation_x = features(validation_rows)
    inspection_x = features(inspection)
    if encoder.cache_misses and not arguments.allow_encode:
        raise RuntimeError(
            f"refusing to re-encode {encoder.cache_misses} clips; "
            "reuse artifacts/starss23-scene-raster/embeddings"
        )
    target_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
        "torch": torch,
    }
    train_y = raster.frame_targets(train_rows, [len(clip) for clip in train_x], **target_arguments)
    validation_y = raster.frame_targets(
        validation_rows,
        [len(clip) for clip in validation_x],
        **target_arguments,
    )
    train_matrix = torch.cat(train_x)
    validation_matrix = torch.cat(validation_x)
    inspection_lengths = [len(clip) for clip in inspection_x]
    inspection_matrix = torch.cat(inspection_x)
    train_targets = torch.cat(train_y)
    validation_targets = torch.cat(validation_y)
    mean = train_matrix.mean(dim=0)
    scale = train_matrix.std(dim=0).clamp_min(1e-5)
    train_matrix = (train_matrix - mean) / scale
    validation_matrix = (validation_matrix - mean) / scale
    inspection_matrix = (inspection_matrix - mean) / scale
    positives = train_targets.sum()
    negatives = train_targets.numel() - positives
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=(negatives / positives.clamp_min(1)).reshape(1)
    )
    head = raster.build_mlp_head(torch)
    raster.assert_head_is_mlp_not_conv_or_gru(head, torch)
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise RuntimeError("encoder thawed before AdamW")
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    best_state = None
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(1, arguments.epochs + 1):
        head.train()
        optimizer.zero_grad()
        train_loss = loss_function(head(train_matrix), train_targets)
        train_loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            validation_loss = loss_function(head(validation_matrix), validation_targets).item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss.item(),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= arguments.patience:
                break
    if best_state is None:
        raise RuntimeError("STARSS23 temporal head training produced no checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    decoder = dict(PREDECLARED_DECODER)
    span_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
    }
    with torch.inference_mode():
        inspection_flat = torch.sigmoid(head(inspection_matrix))
        validation_flat = torch.sigmoid(head(validation_matrix))
    inspection_probabilities = list(inspection_flat.split(inspection_lengths))
    validation_probabilities = list(validation_flat.split([len(clip) for clip in validation_x]))
    references = [row["events"] for row in inspection]
    predictions = raster.decode_spans(
        inspection_probabilities,
        decoder,
        **span_arguments,
    )
    validation_predictions = raster.decode_spans(
        validation_probabilities,
        decoder,
        **span_arguments,
    )
    baseline = whole_clip_predictions(references, duration_ms=DURATION_MS)
    temporal_segment = segment_f1(references, predictions, duration_ms=DURATION_MS)
    baseline_segment = segment_f1(references, baseline, duration_ms=DURATION_MS)
    temporal_collar = collar_event_metrics(references, predictions)
    baseline_collar = collar_event_metrics(references, baseline)
    margin = float(temporal_segment["f1"]) - float(baseline_segment["f1"])
    collar_f1 = float(temporal_collar["f1"])
    gate_passed = should_wire_starss23_timestamps(
        segment_f1=float(temporal_segment["f1"]),
        whole_clip_segment_f1=float(baseline_segment["f1"]),
        collar_f1=collar_f1,
    )
    validation_collar = collar_event_metrics(
        [row["events"] for row in validation_rows],
        validation_predictions,
    )
    gate = {
        "passed": gate_passed,
        "segment_margin_required": SEGMENT_MARGIN_REQUIRED,
        "segment_margin_observed": margin,
        "collar_f1_required": COLLAR_F1_REQUIRED,
        "collar_f1_observed": collar_f1,
        "temporal_segment_f1": float(temporal_segment["f1"]),
        "whole_clip_segment_f1": float(baseline_segment["f1"]),
    }
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": best_state,
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": LABELS,
            "hidden_size": MLP_HIDDEN_SIZE,
            "threshold": decoder["high_threshold"],
            "decoder": {"type": "hysteresis", **decoder},
            "dataset": "starss23",
            "embedding": SENSEVOICE_FRAME_EMBEDDING,
            "frame_hop_ms_approx": encoder.frame_hop_ms,
            "first_frame_center_ms_approx": encoder.first_frame_center_ms,
            "encoder_frozen": True,
            "gate": gate,
        },
        arguments.checkpoint_output,
    )
    checkpoint_sha256 = digest(arguments.checkpoint_output)
    predictions_by_id = {
        row["clip_id"]: prediction
        for row, prediction in zip(inspection, predictions, strict=True)
    }
    selected = select_listenpack_clips(inspection)
    write_fixture_htmls(arguments.listenpack_dir)
    listenpack = write_listen_pack(
        selected,
        predictions_by_id,
        arguments.listenpack_dir,
        copy_audio=arguments.copy_listenpack_audio,
        wired=gate_passed,
    )
    payload = {
        "report_version": "1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "open" if gate_passed else "closed",
        "inspection_evaluations": 1,
        "protocol": PROTOCOL,
        "split_protocol": SPLIT_PROTOCOL,
        "task": "STARSS23 v1.1 first-60s frozen-MLP laughter localization listen pack",
        "headline": (
            "frozen first-60s MLP on 100 ms overlays; "
            f"inspection collar {collar_f1:.4f} vs reported best 0.1395; "
            f"{'WIRE' if gate_passed else 'unwired'}"
        ),
        "label_mapping": {"STARSS23 class 4 laughter": "Attune laugh"},
        "language": "unverified; STARSS23 metadata has no language field",
        "label_status": "human 100 ms activity label; owner-authorized; not second-listener Attune gold",
        "source_label_status": HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD,
        "decoder_lock_note": DECODER_LOCK_NOTE,
        "decoder": dict(PREDECLARED_DECODER),
        "decoder_searched": False,
        "tiling_in_train": False,
        "encoder_unfrozen": False,
        "manifests": {
            "development": {
                "path": str(arguments.development_manifest),
                "sha256": digest(arguments.development_manifest),
            },
            "inspection_test": {
                "path": str(arguments.inspection_manifest),
                "sha256": digest(arguments.inspection_manifest),
            },
        },
        "encoder": encoder.metadata(),
        "encoder_frozen": True,
        "head": {
            "type": "two-layer binary MLP over frozen 512-d acoustic frames",
            "architecture": "mlp",
            "hidden_size": MLP_HIDDEN_SIZE,
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "checkpoint": str(arguments.checkpoint_output),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_committed": False,
            "retrained": True,
        },
        "partitions": {
            "train_clips": len(train_rows),
            "train_events": event_count(train_rows),
            "validation_clips": len(validation_rows),
            "validation_events": event_count(validation_rows),
            "inspection_test_clips": len(inspection),
            "inspection_test_events": event_count(inspection),
            "validation_rooms": sorted(VALIDATION_ROOMS),
            "scene_disjoint_validation": True,
            "file_and_room_disjoint_inspection": True,
        },
        "training": {
            "seed": arguments.seed,
            "epochs_completed": len(history),
            "patience": arguments.patience,
            "loss": "BCEWithLogitsLoss",
            "positive_class_weight": float((negatives / positives.clamp_min(1)).reshape(())),
            "note": (
                "Copies 27df8e1 40-epoch loop (class pos_weight from train frames). "
                "Later notes named that checkpoint 40_epoch_unweighted to contrast "
                "the 73-epoch retry; 27df8e1 source used pos_weight."
            ),
            "history": history,
        },
        "validation_sanity": {
            "decoder": dict(PREDECLARED_DECODER),
            "collar_event_metrics": validation_collar,
            "note": "locked decoder; this score did not select a decoder",
        },
        "inspection_test": {
            "designation": "locked 48-event / 49-clip gold-review pack; never used for fitting or decoder",
            "event_count": event_count(inspection),
            "temporal_head": {
                "segment_f1": temporal_segment,
                "collar_event_metrics": temporal_collar,
            },
            "whole_clip_oracle_tag_baseline": {
                "definition": (
                    "uses each scene's known laughter presence but assigns 0..60000 ms; "
                    "this is deliberately not localization"
                ),
                "segment_f1": baseline_segment,
                "collar_event_metrics": baseline_collar,
            },
        },
        "comparison_to_0.1395": {
            "prior_best": PRIOR_BEST_PASS,
            "this_collar_f1": collar_f1,
            "this_collar_tp_fp_fn": [
                int(temporal_collar["true_positive"]),
                int(temporal_collar["false_positive"]),
                int(temporal_collar["false_negative"]),
            ],
            "this_segment_f1": float(temporal_segment["f1"]),
            "this_whole_clip_segment_f1": float(baseline_segment["f1"]),
            "this_margin": margin,
            "delta_collar_f1": collar_f1 - PRIOR_BEST_COLLAR_F1,
            "beats_reported_best": collar_f1 > PRIOR_BEST_COLLAR_F1,
        },
        "cascade_wiring": {
            **gate,
            "timestamps_wired": gate_passed,
            "wired": gate_passed,
            "policy": (
                "wire STARSS23 laugh timestamps only if collar F1 >= 0.25 "
                "and segment margin >= 0.05"
            ),
            "decision": (
                "wire STARSS23 laugh timing from the frozen first-60s MLP"
                if gate_passed
                else "leave STARSS23 unwired; frozen first-60s MLP did not clear the collar gate"
            ),
        },
        "listen_pack": {
            "path": str(arguments.listenpack_dir),
            "index": str(arguments.listenpack_dir / "index.html"),
            "manifest": str(arguments.listenpack_dir / "listenpack.json"),
            "n_clips": listenpack["n_clips"],
            "caption": LISTENPACK_CAPTION,
            "local_paths_only": True,
            "public_tunnel": False,
        },
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "embedding_cache_hits": encoder.cache_hits,
            "embedding_cache_misses": encoder.cache_misses,
        },
        "limitations": [
            "Labels are STARSS23 100 ms overlays, owner-authorized, not second-listener gold.",
            "Only STARSS23 laughter maps honestly to the current Attune event ontology.",
            "Speech language is unverified and speech classes are not training targets.",
            "The listen pack is a research tool; product infer stays unwired unless the gate clears.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {arguments.output}")
    print(
        "inspection collar "
        f"{collar_f1:.4f} ({temporal_collar['true_positive']}/"
        f"{temporal_collar['false_positive']}/{temporal_collar['false_negative']}) "
        f"segment {float(temporal_segment['f1']):.4f} margin {margin:.4f} "
        f"wired={gate_passed}"
    )
    print(f"checkpoint sha256 {checkpoint_sha256}")
    print(f"listen pack {arguments.listenpack_dir / 'index.html'}")


if __name__ == "__main__":
    main()
