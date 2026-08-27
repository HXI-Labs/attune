"""STARSS23 first-60s gold-review pack: filter, hash check, ledger, HTML."""

from __future__ import annotations

import hashlib
import json
import math
import wave
from array import array
from collections.abc import Iterable, Sequence
from html import escape
from pathlib import Path
from typing import Any

from attune.data.gold_review import (
    HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD,
    GoldReviewRecord,
)

FIRST_60S_WINDOW_START_MS = 0
EXPECTED_FIRST_60S_CLIPS = 49
EXPECTED_FIRST_60S_LAUGH_EVENTS = 48
EXPECTED_FIRST_60S_TRUE_NEGATIVES = 29
EXPECTED_WINDOW_END_TRUNCATED_LAUGHS = 3
# Source 100 ms overlays kept as-is; flagged incomplete even when
# clipped_spanning_event_count is 0 (PR 22 first-tile keep-truncated behaviour).
WINDOW_END_TRUNCATED_LAUGHS = (
    ("starss23-scene-inspection_test-fold4_room24_mix006-w000000", 59400, 60000),
    ("starss23-scene-inspection_test-fold4_room16_mix010-w000000", 51900, 60000),
    ("starss23-scene-inspection_test-fold4_room8_mix002-w000000", 59500, 60000),
)
PACK_RELATIVE_PATH = Path("data/manifests/starss23-gold-review-pack.jsonl")
PROVENANCE_RELATIVE_PATH = Path("data/manifests/starss23-gold-review-pack.provenance.json")
DEFAULT_HTML_RELATIVE_PATH = Path("artifacts/gold-review/starss23-first60s")
DEFAULT_LEDGER_NAME = "ledger.jsonl"
SOURCE_INSPECTION_MANIFEST = "data/manifests/starss23-scene-raster-inspection.jsonl"
SOURCE_PR = 22
SOURCE_BRANCH = "cursor/starss23-scene-timing"


class AudioHashMismatchError(ValueError):
    """Local audio exists but does not match the committed pack hash."""


class Starss23ReviewStatusError(ValueError):
    """STARSS23 ledger rows must use the explicit non-gold activity status."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def is_first_60s_row(row: dict[str, Any]) -> bool:
    return row.get("source_window_start_ms") == FIRST_60S_WINDOW_START_MS


def first_60s_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if is_first_60s_row(row)]


def laugh_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in row.get("events", []) if event.get("label") == "laugh"]


def laugh_event_count(rows: Sequence[dict[str, Any]]) -> int:
    return sum(len(laugh_events(row)) for row in rows)


def is_true_negative(row: dict[str, Any]) -> bool:
    return len(laugh_events(row)) == 0


def true_negative_count(rows: Sequence[dict[str, Any]]) -> int:
    return sum(1 for row in rows if is_true_negative(row))


def source_span_id(event: dict[str, Any]) -> str:
    return f"laugh@{event['start_ms']}-{event['end_ms']}"


def event_is_truncated_at_window_end(event: dict[str, Any], duration_ms: int) -> bool:
    """Window-end events are truncated even if clipped_spanning_event_count is 0."""
    end_ms = event.get("end_ms")
    return end_ms is not None and int(end_ms) >= int(duration_ms)


def annotate_source_event(event: dict[str, Any], duration_ms: int) -> dict[str, Any]:
    """Flag incomplete window-end events without rewriting source start/end."""
    annotated = dict(event)
    truncated = event_is_truncated_at_window_end(event, duration_ms)
    annotated["start_ms"] = event["start_ms"]
    annotated["end_ms"] = event["end_ms"]
    annotated["truncated_at_window_end"] = truncated
    annotated["incomplete"] = truncated
    annotated["collar_eligible"] = not truncated
    return annotated


def annotate_pack_row(row: dict[str, Any]) -> dict[str, Any]:
    packed_row = dict(row)
    duration_ms = int(packed_row["duration_ms"])
    packed_row["events"] = [
        annotate_source_event(event, duration_ms) for event in row.get("events", [])
    ]
    packed_row["window_end_truncated_event_count"] = sum(
        1
        for event in packed_row["events"]
        if event.get("label") == "laugh" and event.get("truncated_at_window_end")
    )
    packed_row["true_negative"] = is_true_negative(packed_row)
    packed_row["review_pack"] = "starss23-first60s"
    packed_row["source_label_status"] = HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD
    return packed_row


def pack_counts(rows: Sequence[dict[str, Any]]) -> tuple[int, int]:
    return len(rows), laugh_event_count(rows)


def assert_expected_pack_counts(rows: Sequence[dict[str, Any]]) -> None:
    clips, events = pack_counts(rows)
    if clips != EXPECTED_FIRST_60S_CLIPS or events != EXPECTED_FIRST_60S_LAUGH_EVENTS:
        raise ValueError(
            "STARSS23 gold-review pack must be exactly "
            f"{EXPECTED_FIRST_60S_CLIPS} first-60s clips / "
            f"{EXPECTED_FIRST_60S_LAUGH_EVENTS} laugh events, "
            f"got {clips}/{events}"
        )
    if any(not is_first_60s_row(row) for row in rows):
        raise ValueError("STARSS23 gold-review pack must exclude later tiles")
    negatives = true_negative_count(rows)
    if negatives != EXPECTED_FIRST_60S_TRUE_NEGATIVES:
        raise ValueError(
            "STARSS23 gold-review pack must keep "
            f"{EXPECTED_FIRST_60S_TRUE_NEGATIVES} zero-laugh true negatives, "
            f"got {negatives}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_audio_sha256(path: Path, expected: str) -> str:
    digest = sha256_file(path)
    if digest != expected:
        raise AudioHashMismatchError(
            f"sha256 mismatch for {path}: expected {expected}, got {digest}"
        )
    return digest


def build_pack_rows(inspection_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    packed = [annotate_pack_row(row) for row in first_60s_rows(inspection_rows)]
    assert_expected_pack_counts(packed)
    return packed


def subset_content_sha256(rows: Sequence[dict[str, Any]]) -> str:
    payload = "".join(sorted(row["clip_id"] + "\0" + row["sha256"] + "\n" for row in rows))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_cache_dirs(repo_root: Path) -> list[Path]:
    candidates = [
        repo_root / "data/raw/starss23-scene-raster/inspection_test",
        repo_root / "data/raw/starss23-scene-raster",
        repo_root.parent / "attune/data/raw/starss23-scene-raster/inspection_test",
        repo_root.parent / "attune/data/raw/starss23-scene-raster",
        repo_root / "data/raw/starss23-scene-raster-tiled/inspection_test",
        repo_root / "data/raw/starss23-scene-raster-tiled",
        repo_root.parent / "attune/data/raw/starss23-scene-raster-tiled/inspection_test",
        repo_root.parent / "attune/data/raw/starss23-scene-raster-tiled",
    ]
    return [path for path in candidates if path.exists()]


def _wavs_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.wav") if path.is_file())


def resolve_pack_audio(
    rows: Sequence[dict[str, Any]],
    cache_dirs: Sequence[Path],
) -> tuple[dict[str, Path], list[str], list[str]]:
    """Return hash-matched paths, missing clip ids, and mismatch messages.

    A file that exists at a row cache_path with the wrong hash fails closed.
    """
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    mismatches: list[str] = []
    hash_index: dict[str, Path] = {}
    for cache_dir in cache_dirs:
        for wav in _wavs_under(cache_dir):
            digest = sha256_file(wav)
            hash_index.setdefault(digest, wav)

    for row in rows:
        clip_id = row["clip_id"]
        expected = row["sha256"]
        explicit_hits: list[Path] = []
        relative = Path(row["cache_path"])
        for cache_dir in cache_dirs:
            for candidate in (
                cache_dir / relative,
                cache_dir / relative.name,
            ):
                if candidate.is_file() and candidate not in explicit_hits:
                    explicit_hits.append(candidate)
        failed_explicit = False
        for candidate in explicit_hits:
            digest = sha256_file(candidate)
            if digest != expected:
                mismatches.append(f"{clip_id}: {candidate} expected {expected} got {digest}")
                failed_explicit = True
                break
            resolved[clip_id] = candidate
            break
        if failed_explicit:
            continue
        if clip_id in resolved:
            continue
        indexed = hash_index.get(expected)
        if indexed is not None:
            resolved[clip_id] = indexed
        else:
            missing.append(clip_id)

    if mismatches:
        raise AudioHashMismatchError(
            "sha256 mismatch; refusing to invent audio:\n" + "\n".join(mismatches)
        )
    return resolved, missing, mismatches


def validate_starss23_review(record: GoldReviewRecord, pack_row: dict[str, Any]) -> None:
    if record.source_label_status != HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD:
        raise Starss23ReviewStatusError(
            "STARSS23 reviews must use source_label_status="
            f"{HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD}, not "
            f"{record.source_label_status}"
        )
    if record.dry_run_fixture:
        raise Starss23ReviewStatusError("STARSS23 reviews cannot be dry-run fixtures")
    if record.clip_id != pack_row["clip_id"]:
        raise Starss23ReviewStatusError("review clip_id does not match the pack row")
    if record.audio_sha256 != pack_row["sha256"]:
        raise AudioHashMismatchError(
            f"review audio_sha256 {record.audio_sha256} != pack {pack_row['sha256']}"
        )
    if record.audio_duration_ms != pack_row["duration_ms"]:
        raise Starss23ReviewStatusError("review duration does not match the pack row")
    if record.transcript.decision != "not_reviewable":
        raise Starss23ReviewStatusError(
            "STARSS23 reviews auto-fill transcript as not_reviewable; consent is "
            "not independently verified and overlapping speech is not transcribed"
        )
    if record.affect.decision != "not_reviewable":
        raise Starss23ReviewStatusError("STARSS23 reviews auto-fill affect as not_reviewable")
    for span in record.spans:
        if span.decision == "accept":
            raise Starss23ReviewStatusError(
                "STARSS23 100 ms source spans cannot be accepted as onset gold; "
                "use reject, retime, or add at free millisecond resolution"
            )


def append_ledger(path: Path, record: GoldReviewRecord, pack_row: dict[str, Any]) -> None:
    validate_starss23_review(record, pack_row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def load_ledger(path: Path) -> list[GoldReviewRecord]:
    if not path.is_file():
        return []
    return [
        GoldReviewRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wav_peak_envelope(path: Path, buckets: int = 1200) -> list[float]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
    if width != 2:
        raise ValueError(f"expected PCM16 wav, got sample width {width} at {path}")
    samples = array("h")
    samples.frombytes(raw)
    if channels > 1:
        samples = array(
            "h",
            (
                round(sum(samples[index : index + channels]) / channels)
                for index in range(0, len(samples), channels)
            ),
        )
    if not samples:
        return [0.0] * buckets
    bucket_size = max(1, math.ceil(len(samples) / buckets))
    peaks: list[float] = []
    for start in range(0, len(samples), bucket_size):
        chunk = samples[start : start + bucket_size]
        peak = max(abs(value) for value in chunk) / 32767.0
        peaks.append(round(min(1.0, peak), 4))
        if len(peaks) == buckets:
            break
    while len(peaks) < buckets:
        peaks.append(0.0)
    return peaks


def copy_verified_wav(source: Path, destination: Path, expected: str) -> None:
    verify_audio_sha256(source, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    verify_audio_sha256(destination, expected)


def clip_payload(
    row: dict[str, Any],
    *,
    audio_href: str | None,
    peaks: list[float] | None,
) -> dict[str, Any]:
    events = []
    for event in laugh_events(row):
        events.append(
            {
                "source_label": source_span_id(event),
                "label": event["label"],
                "start_ms": event["start_ms"],
                "end_ms": event["end_ms"],
                "source_class": event.get("source_class"),
                "source_indices": event.get("source_indices"),
                "truncated_at_window_end": bool(event.get("truncated_at_window_end")),
                "incomplete": bool(event.get("incomplete") or event.get("truncated_at_window_end")),
                "collar_eligible": bool(event.get("collar_eligible", True))
                and not bool(event.get("truncated_at_window_end")),
            }
        )
    return {
        "clip_id": row["clip_id"],
        "sha256": row["sha256"],
        "duration_ms": row["duration_ms"],
        "source_recording": row.get("source_recording"),
        "room": row.get("room"),
        "official_partition": row.get("official_partition"),
        "language": row.get("language"),
        "label_status": row.get("label_status"),
        "source_label_status": HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD,
        "downmix": row.get("downmix"),
        "attribution": row.get("attribution"),
        "licence": row.get("licence"),
        "audio_href": audio_href,
        "peaks": peaks,
        "events": events,
        "true_negative": is_true_negative(row),
        "gold": False,
    }


def _style() -> str:
    return """
:root { color-scheme: dark; --bg:#10141c; --panel:#1a2130; --ink:#e8eef8; --muted:#9aa8bc;
        --line:#2a3548; --accent:#7db4ff; --warn:#f3c16b; --bad:#f08a8a; --ok:#8fd6a8; }
* { box-sizing: border-box; }
body { margin:0; font: 15px/1.45 ui-sans-serif, system-ui, sans-serif; background:var(--bg);
       color:var(--ink); }
a { color:var(--accent); }
header, main { max-width: 1100px; margin: 0 auto; padding: 1.2rem; }
.banner { background:#3a2a12; color:var(--warn); padding:.75rem 1rem; border:1px solid #6b5424;
          border-radius:8px; margin-bottom:1rem; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:1rem;
         margin:1rem 0; }
table { width:100%; border-collapse:collapse; }
th, td { text-align:left; padding:.4rem .5rem; border-bottom:1px solid var(--line); }
.muted { color:var(--muted); }
button, select, input, textarea { font: inherit; }
button { background:var(--accent); color:#0b1220; border:0; border-radius:6px; padding:.4rem .8rem;
         cursor:pointer; }
button.secondary { background:transparent; color:var(--ink); border:1px solid var(--line); }
label { display:block; margin:.4rem 0 .2rem; }
input[type=text], input[type=number], textarea, select { width:100%; background:#0f1520;
  color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:.35rem .5rem; }
.span-card { border:1px solid var(--line); border-radius:8px; padding:.7rem; margin:.6rem 0; }
canvas { width:100%; height:120px; background:#0b1018; border-radius:8px; cursor:crosshair; }
audio { width:100%; margin:.6rem 0; }
.grid { display:grid; grid-template-columns: 1fr 1fr; gap:.6rem; }
@media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
""".strip()


def _app_js() -> str:
    return r"""
function $(id) { return document.getElementById(id); }
function drawWave(canvas, peaks, durationMs, events, selection) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.clientWidth * devicePixelRatio;
  const h = canvas.height = canvas.clientHeight * devicePixelRatio;
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
  events.forEach(ev => mark(ev.start_ms, ev.end_ms, "rgba(243,193,107,0.28)"));
  if (selection && selection.start_ms != null && selection.end_ms != null) {
    mark(selection.start_ms, selection.end_ms, "rgba(143,214,168,0.35)");
  }
}
function isoNow() { return new Date().toISOString().replace(/\\.\\d{3}Z$/, "Z"); }
function collectRecord(clip) {
  const spans = [];
  document.querySelectorAll("[data-span]").forEach(card => {
    const decision = card.querySelector("[name=decision]").value;
    if (!decision || decision === "accept") return;
    const sourceLabel = card.dataset.sourceLabel || null;
    const reviewed = card.querySelector("[name=reviewed_label]").value || null;
    const start = card.querySelector("[name=start_ms]");
    const end = card.querySelector("[name=end_ms]");
    let channel = "event";
    if (reviewed === "laughing_speech") channel = "style";
    if (reviewed === "laugh") channel = "event";
    const span = { channel, source_label: sourceLabel || null,
                   decision, reviewed_label: reviewed, start_ms: null, end_ms: null };
    if (decision === "retime" || decision === "add") {
      span.start_ms = start.value === "" ? null : Number(start.value);
      span.end_ms = end.value === "" ? null : Number(end.value);
    }
    if (decision === "add" && !span.source_label && span.start_ms != null && span.end_ms != null) {
      span.source_label = "added@" + span.start_ms + "-" + span.end_ms;
    }
    spans.push(span);
  });
  return {
    schema_version: "1.0",
    clip_id: clip.clip_id,
    audio_sha256: clip.sha256,
    audio_duration_ms: clip.duration_ms,
    reviewer: $("reviewer").value.trim(),
    reviewed_at_utc: isoNow(),
    source_label_status: "human_100ms_activity_not_attune_gold",
    transcript: { decision: "not_reviewable", corrected_text: null },
    affect: { decision: "not_reviewable", reviewed_label: null },
    spans,
    notes: $("notes").value,
    dry_run_fixture: false
  };
}
function boot(clip) {
  const canvas = $("wave");
  const selection = { start_ms: null, end_ms: null };
  if (canvas && clip.peaks) {
    const redraw = () => drawWave(canvas, clip.peaks, clip.duration_ms, clip.events, selection);
    redraw();
    window.addEventListener("resize", redraw);
    canvas.addEventListener("click", ev => {
      const rect = canvas.getBoundingClientRect();
      // Free millisecond resolution; never snap to the 100 ms source grid.
      const ms = Math.round((ev.clientX - rect.left) / rect.width * clip.duration_ms);
      if (selection.start_ms == null || (selection.end_ms != null)) {
        selection.start_ms = ms; selection.end_ms = null;
      } else {
        selection.end_ms = Math.max(ms, selection.start_ms);
      }
      $("click_start").value = selection.start_ms ?? "";
      $("click_end").value = selection.end_ms ?? "";
      redraw();
    });
  }
  $("save").addEventListener("click", async () => {
    const record = collectRecord(clip);
    if (!record.reviewer) { alert("Reviewer id is required."); return; }
    $("record_json").value = JSON.stringify(record, null, 2);
    const blob = new Blob([JSON.stringify(record) + "\\n"], {type: "application/jsonl"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = record.clip_id + "." + record.reviewer + ".jsonl";
    a.click();
    if (location.protocol.startsWith("http")) {
      const response = await fetch("/append", {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify(record)
      });
      const body = await response.json();
      $("save_status").textContent = response.ok
        ? "Appended to local ledger."
        : ("Ledger error: " + (body.error || response.status));
    } else {
      $("save_status").textContent =
        "Downloaded JSONL row. Serve with the review script to append the gitignored ledger.";
    }
  });
}
""".strip()


def _label_definitions_html() -> str:
    return """
<div class="panel" id="label-definitions">
  <h2>Label definitions</h2>
  <ul>
    <li><code>laugh</code>: discrete non-speech laughter event.</li>
    <li><code>laughing_speech</code>: laughter that modifies speech
        (a style, not a discrete event).</li>
    <li>Onset = first voiced burst. Offset = last voiced frame.</li>
    <li>Split into separate events when a silent/unvoiced gap is >= 300 ms.</li>
    <li>Downmix-audible only: label only what is audible in the
        mean-of-4 16 kHz downmix.</li>
    <li>>= 2 independent reviewers plus adjudication before any gold claim.</li>
    <li>Do not compare gold collar to 0.1395
        (failed frozen-MLP ceiling, not a review target).</li>
    <li>Source 100 ms spans are overlays only. Do not accept them as onset gold.
        Default action is retime at free millisecond resolution. Never snap to 100 ms.</li>
    <li>Window-end truncated events are incomplete and collar-ineligible.
        Do not gold clip-end as offset.</li>
    <li>Transcript and affect are not reviewer tasks. STARSS23 consent is not independently
        verified; do not transcribe overlapping speech. Ledger fields auto-fill
        <code>not_reviewable</code>.</li>
  </ul>
</div>
""".strip()


def render_index(rows: Sequence[dict[str, Any]], missing: Sequence[str]) -> str:
    missing_set = set(missing)
    body_rows = []
    for row in rows:
        n_events = len(laugh_events(row))
        audio_state = "missing wav" if row["clip_id"] in missing_set else "hash-matched"
        href = f"clips/{escape(row['clip_id'])}.html"
        truncated = any(
            event.get("truncated_at_window_end")
            or event_is_truncated_at_window_end(event, int(row["duration_ms"]))
            for event in laugh_events(row)
        )
        if is_true_negative(row):
            role = "true negative"
        elif truncated:
            role = "source laugh overlay; truncated window-end"
        else:
            role = "source laugh overlay"
        body_rows.append(
            "<tr>"
            f"<td><a href='{href}'>{escape(row['clip_id'])}</a></td>"
            f"<td>{escape(str(row.get('room', '')))}</td>"
            f"<td>{n_events}</td>"
            f"<td>{escape(role)}</td>"
            f"<td>{escape(audio_state)}</td>"
            "</tr>"
        )

    missing_banner = ""
    if missing:
        missing_banner = (
            f"<div class='banner'>Missing wavs for {len(missing)} clip(s). "
            "Pages still exist; audio is omitted. Do not invent audio.</div>"
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>STARSS23 first-60s gold review</title>
<style>{_style()}</style></head>
<body>
<header>
  <div class="banner"><strong>Gold gate CLOSED.</strong> STARSS23 human 100 ms activity
  is not Attune gold. One pass is reviewer-labelled. Independent second pass and
  adjudication are required. Never commit wavs.</div>
  <h1>STARSS23 first-60s gold-review pack</h1>
  <p class="muted">{EXPECTED_FIRST_60S_CLIPS} clips /
  {EXPECTED_FIRST_60S_LAUGH_EVENTS} source laugh events /
  {EXPECTED_FIRST_60S_TRUE_NEGATIVES} true-negative clips.
  Filter: <code>source_window_start_ms == 0</code>. Later tiles excluded.
  Source status: <code>{HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD}</code>.</p>
  {missing_banner}
  <p>Privacy: public MIT natural scenes with identifiable speech. STARSS23 consent
  is not independently verified. Do not transcribe overlapping speech.
  Review stays local. Do not upload participant audio.
  See <code>docs/gold-review-starss23.md</code>.</p>
</header>
{_label_definitions_html()}
<main class="panel">
<table>
<thead><tr><th>clip</th><th>room</th><th>source laughs</th>
<th>pack role</th><th>audio</th></tr></thead>
<tbody>{"".join(body_rows)}</tbody>
</table>
</main>
</body></html>
"""


def render_clip(payload: dict[str, Any]) -> str:
    events_html = []
    for event in payload["events"]:
        truncated = bool(event.get("truncated_at_window_end") or event.get("incomplete"))
        trunc_banner = (
            "<p class='banner'>Incomplete window-end truncation. Collar-ineligible. "
            "Do not gold clip-end as offset.</p>"
            if truncated
            else ""
        )
        events_html.append(
            f"""
<div class="span-card" data-span data-kind="event"
     data-source-label="{escape(event["source_label"])}">
  <strong>Source class-4 overlay</strong>
  {trunc_banner}
  <p class="muted">{escape(event["source_label"])} · {event["start_ms"]}–{event["end_ms"]} ms
  (STARSS23 100 ms activity overlay, merged). Overlay only — not onset gold.
  Reject, retime, or add at free millisecond resolution. Default retime. Never snap to 100 ms.</p>
  <div class="grid">
    <label>Decision
      <select name="decision">
        <option value="reject">reject</option>
        <option value="retime" selected>retime</option>
      </select>
    </label>
    <label>Reviewed label
      <select name="reviewed_label">
        <option value="laugh">laugh</option>
        <option value="laughing_speech">laughing_speech</option>
      </select>
    </label>
    <label>Onset ms (retime)
      <input type="number" name="start_ms" min="0" step="1"
        max="{payload["duration_ms"]}"
        placeholder="overlay {event["start_ms"]}; free ms, never snap">
    </label>
    <label>Offset ms (retime)
      <input type="number" name="end_ms" min="0" step="1"
        max="{payload["duration_ms"]}"
        placeholder="overlay {event["end_ms"]}; free ms, never snap">
    </label>
  </div>
</div>
"""
        )
    audio = (
        f"<audio controls src='{escape(payload['audio_href'])}'></audio>"
        if payload.get("audio_href")
        else "<p class='banner'>Audio missing; page is review-structure only.</p>"
    )
    clip_json = json.dumps(payload, ensure_ascii=True)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{escape(payload["clip_id"])}</title>
<style>{_style()}</style></head>
<body>
<header>
  <p><a href="../index.html">← pack index</a></p>
  <div class="banner"><strong>Not gold.</strong> Source 100 ms spans are overlays only.
  Do not accept them as onset gold. Default retime at free millisecond resolution.
  Never snap to 100 ms. Attune model scores are not provided and must not be treated as answers.
  Transcript and affect are not reviewer tasks and auto-fill <code>not_reviewable</code>.
  STARSS23 consent is not independently verified; do not transcribe overlapping speech.</div>
  <h1>{escape(payload["clip_id"])}</h1>
  <p class="muted">{escape(str(payload.get("source_recording")))} ·
  {escape(str(payload.get("room")))} ·
  {payload["duration_ms"]} ms · sha256 {escape(payload["sha256"][:16])}… ·
  {escape(str(payload.get("downmix")))}</p>
</header>
{_label_definitions_html()}
<main>
  <div class="panel">
    {audio}
    <canvas id="wave"></canvas>
    <p class="muted">Orange overlays are STARSS23 class-4 100 ms activity. Click once for onset,
    again for offset at free millisecond resolution (never snap to 100 ms).
    Onset = first voiced burst. Offset = last voiced frame.</p>
    <div class="grid">
      <label>Clicked onset ms <input id="click_start" readonly></label>
      <label>Clicked offset ms <input id="click_end" readonly></label>
    </div>
  </div>
  <div class="panel">
    <label>Reviewer id <input id="reviewer" type="text" autocomplete="username"></label>
    <p class="muted">Transcript and affect are hidden reviewer tasks. Ledger fields
    auto-fill <code>not_reviewable</code>. STARSS23 consent is not independently
    verified; do not transcribe overlapping speech.</p>
    {"".join(events_html)}
    <div class="span-card" data-span data-kind="event" data-source-label="">
      <strong>Add missing audible laugh / laughing_speech</strong>
      <div class="grid">
        <label>Decision
          <select name="decision">
            <option value="">none</option>
            <option value="add">add</option>
          </select>
        </label>
        <label>Reviewed label
          <select name="reviewed_label">
            <option value="laugh">laugh</option>
            <option value="laughing_speech">laughing_speech</option>
          </select>
        </label>
        <label>Onset ms <input type="number" name="start_ms" min="0" step="1"
          max="{payload["duration_ms"]}"></label>
        <label>Offset ms <input type="number" name="end_ms" min="0" step="1"
          max="{payload["duration_ms"]}"></label>
      </div>
    </div>
    <label>Notes <textarea id="notes" rows="3"></textarea></label>
    <p>
      <button id="save" type="button">Write review row</button>
      <button class="secondary" type="button" onclick="history.back()">Back</button>
      <span id="save_status" class="muted"></span>
    </p>
    <label>Record JSON <textarea id="record_json" rows="10" readonly></textarea></label>
  </div>
</main>
<script>{_app_js()}</script>
<script>boot({clip_json});</script>
</body></html>
"""


def write_html_pack(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    resolved: dict[str, Path],
    missing: Sequence[str],
    *,
    copy_audio: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    audio_dir = output_dir / "audio"
    clips_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in rows:
        clip_id = row["clip_id"]
        audio_href = None
        peaks = None
        source = resolved.get(clip_id)
        if source is not None:
            destination = audio_dir / f"{clip_id}.wav"
            if copy_audio:
                copy_verified_wav(source, destination, row["sha256"])
                copied += 1
                peaks = wav_peak_envelope(destination)
                audio_href = f"../audio/{clip_id}.wav"
            else:
                verify_audio_sha256(source, row["sha256"])
                peaks = wav_peak_envelope(source)
                audio_href = source.resolve().as_uri()
        payload = clip_payload(row, audio_href=audio_href, peaks=peaks)
        (clips_dir / f"{clip_id}.html").write_text(render_clip(payload), encoding="utf-8")
    (output_dir / "index.html").write_text(render_index(rows, missing), encoding="utf-8")
    summary = {
        "clips": len(rows),
        "laugh_events": laugh_event_count(rows),
        "copied_wavs": copied,
        "missing": list(missing),
        "true_negatives": true_negative_count(rows),
        "window_end_truncated_events": sum(
            row.get("window_end_truncated_event_count", 0) for row in rows
        ),
        "gold": False,
        "source_label_status": HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD,
        "index": str(output_dir / "index.html"),
        "ledger": str(output_dir / DEFAULT_LEDGER_NAME),
    }
    (output_dir / "pack-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def pack_provenance(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest": str(PACK_RELATIVE_PATH),
        "row_count": len(rows),
        "laugh_event_count": laugh_event_count(rows),
        "true_negative_count": true_negative_count(rows),
        "window_end_truncated_event_count": sum(
            row.get("window_end_truncated_event_count", 0) for row in rows
        ),
        "filter": "source_window_start_ms == 0",
        "excluded": "later 60s tiles from the failed tiled train protocol",
        "source_span_policy": (
            "100 ms source start/end kept as overlays; never overwritten. "
            "Review may reject, retime, or add at free millisecond resolution; "
            "accept is not onset gold. Window-end events are incomplete and "
            "collar-ineligible even if clipped_spanning_event_count is 0."
        ),
        "derived_from": {
            "pull_request": SOURCE_PR,
            "branch": SOURCE_BRANCH,
            "source_manifest": SOURCE_INSPECTION_MANIFEST,
            "note": (
                "First-60s rows only. PR 22 is not merged. This pack is not "
                "Attune gold and does not wire STARSS23 into training."
            ),
        },
        "dataset": "STARSS23 v1.1 development set",
        "record": "https://zenodo.org/records/7880637",
        "doi": "10.5281/zenodo.7880637",
        "licence": "MIT",
        "language": "unverified",
        "annotation": "human 100 ms activity, class 4 laughter; laugh vs laughing_speech unsplit",
        "label_status": "STARSS23 human activity label; not reviewed Attune gold",
        "source_label_status": HUMAN_100MS_ACTIVITY_NOT_ATTUNE_GOLD,
        "downmix": "mean_of_4_tetrahedral_mic",
        "duration_ms": 60000,
        "sample_rate_hz": 16000,
        "audio_committed": False,
        "gold": False,
        "subset_content_sha256": subset_content_sha256(rows),
        "hash_algorithm": (
            "SHA-256 over UTF-8 lines clip_id + NUL + file_sha256 + LF, sorted by clip_id"
        ),
        "privacy": (
            "Public MIT natural scenes with identifiable speech. STARSS23 consent "
            "is not independently verified. Do not transcribe overlapping speech. "
            "Review stays local. Never commit wavs. Never upload participant audio."
        ),
        "attribution": (
            "STARSS23 by Politis, Shimada, Sudarsanam, Hakala, Takahashi, Krause, "
            "Adavanne, Koyama, Uchida, Mitsufuji, Virtanen, and collaborators."
        ),
    }
