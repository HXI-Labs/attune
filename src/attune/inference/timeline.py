"""Playable HTML timeline for one validated Attune output.

JSON remains authoritative. This page is a display projection: it must not
invent word times or treat utterance-scoped 0..duration bars as localization.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Sequence

from attune.schema.output import AttuneOutput, EventLabel, VocalEvent, VocalStyle

FRAME_EVENT_LABELS = frozenset(
    {EventLabel.LAUGH.value, EventLabel.COUGH.value, EventLabel.THROAT_CLEAR.value}
)
FRAME_LOCAL = "frame-local"
UTTERANCE_SCOPE = "utterance-scope"


def classify_span(
    *,
    channel: str,
    label: str,
    start_ms: int,
    end_ms: int,
    duration_ms: int,
) -> str:
    """Label a span from geometry. Whole-clip bounds are never localization."""
    del channel
    whole_clip = start_ms == 0 and end_ms == duration_ms
    if label in FRAME_EVENT_LABELS and not whole_clip:
        return FRAME_LOCAL
    return UTTERANCE_SCOPE


def render_demo_html(
    output: AttuneOutput,
    *,
    audio_src: str | None = None,
    dcase_head_configured: bool = False,
    fixture: bool = False,
    notes: Sequence[str] = (),
) -> str:
    """Render a self-contained player. Audio is optional and never inlined."""
    duration_ms = output.audio.duration_ms
    events = [_span_view("event", event, duration_ms) for event in output.events]
    styles = [_span_view("style", style, duration_ms) for style in output.styles]
    words = [
        {
            "id": word.id,
            "text": word.text,
            "start_ms": word.start_ms,
            "end_ms": word.end_ms,
        }
        for word in output.transcript.words
    ]
    frame_count = sum(1 for item in events if item["kind"] == FRAME_LOCAL)
    banners = [
        *notes,
        *_banners(
            fixture=fixture,
            dcase_head_configured=dcase_head_configured,
            frame_count=frame_count,
            word_count=len(words),
        ),
    ]
    affect = output.affect
    affect_label = (
        "abstain"
        if affect.abstain
        else (affect.top_label.value if affect.top_label is not None else "unknown")
    )
    distribution_html = _affect_distribution_html(affect)
    payload = {
        "duration_ms": duration_ms,
        "events": events,
        "styles": styles,
        "words": words,
    }
    audio_block = (
        f'<audio controls preload="metadata" src="{html.escape(audio_src, quote=True)}"></audio>'
        if audio_src
        else '<p class="note">No local WAV path was supplied; the timeline still shows spans.</p>'
    )
    word_row = _word_row(words, duration_ms)
    frame_events = [item for item in events if item["kind"] == FRAME_LOCAL]
    utterance_events = [item for item in events if item["kind"] == UTTERANCE_SCOPE]
    event_row = _track_row("Frame-local events", frame_events, duration_ms) + _track_row(
        "Utterance-scope events (not localization)",
        utterance_events,
        duration_ms,
    )
    style_row = _track_row("Utterance-scope styles (not localization)", styles, duration_ms)
    banner_html = "".join(f'<p class="banner">{html.escape(item)}</p>' for item in banners)
    transcript = (
        html.escape(output.transcript.text) or "<span class='muted'>(empty transcript)</span>"
    )
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Attune cascade demo</title>
  <style>
    :root {{ font-family: ui-sans-serif, system-ui, sans-serif; color: #1f2328; }}
    body {{ margin: 1.5rem auto; max-width: 920px; line-height: 1.45; }}
    h1 {{ font-size: 1.25rem; margin-bottom: 0.3rem; }}
    .muted, .note {{ color: #656d76; }}
    .banner {{ background: #fff8c5; border: 1px solid #d4a72c; padding: 0.6rem 0.75rem; }}
    audio {{ width: 100%; margin: 0.75rem 0; }}
    .legend span {{ display: inline-block; margin-right: 1rem; }}
    .swatch {{ display: inline-block; width: 0.9rem; height: 0.9rem;
      vertical-align: -0.1rem; margin-right: 0.3rem; }}
    .swatch.frame-local {{ background: #1f6feb; }}
    .swatch.utterance-scope {{
      background: repeating-linear-gradient(135deg, #d0d7de 0 5px, #f6f8fa 5px 10px);
      border: 1px dashed #656d76;
    }}
    .track {{ margin: 0.85rem 0 1.1rem; }}
    .track h2 {{ font-size: 0.85rem; text-transform: uppercase;
      letter-spacing: 0.04em; margin: 0 0 0.35rem; }}
    .rail {{ position: relative; height: 2.4rem; background: #f6f8fa; border: 1px solid #d0d7de; }}
    .span {{
      position: absolute; top: 0.25rem; bottom: 0.25rem; overflow: hidden;
      font-size: 0.75rem; padding: 0.15rem 0.35rem; box-sizing: border-box; white-space: nowrap;
    }}
    .span.frame-local {{ background: #1f6feb; color: #fff; }}
    .span.utterance-scope {{
      background: repeating-linear-gradient(135deg, #d0d7de 0 6px, #eaeef2 6px 12px);
      border: 1px dashed #656d76; color: #1f2328;
    }}
    .span.word {{ background: #ddf4ff; border: 1px solid #54aeff; color: #0a3069; }}
    .playhead {{
      position: absolute; top: 0; bottom: 0; width: 2px; background: #cf222e; pointer-events: none;
    }}
    .empty {{ color: #656d76; font-size: 0.85rem; padding: 0.45rem; }}
    dl.meta {{ display: grid; grid-template-columns: 10rem 1fr; gap: 0.2rem 0.75rem; }}
    code {{ font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>Attune cascade demo</h1>
  <p class="muted">Authoritative JSON is separate. This page only projects validated spans.</p>
  {banner_html}
  {audio_block}
  <p class="legend">
    <span><i class="swatch frame-local"></i>Frame-local laugh/cough/throat_clear</span>
    <span><i class="swatch utterance-scope"></i>Utterance scope — <em>not</em> localization</span>
  </p>
  {word_row}
  {event_row}
  {style_row}
  <h2>Transcript</h2>
  <p>{transcript}</p>
  <h2>Affect</h2>
  <dl class="meta">
    <dt>top label</dt><dd>{html.escape(affect_label)}</dd>
    <dt>abstain</dt><dd>{str(affect.abstain).lower()}</dd>
    <dt>top confidence</dt><dd>{affect.top_label_confidence:.4f}</dd>
    <dt>distribution</dt><dd>{distribution_html}</dd>
    <dt>span</dt><dd>{affect.start_ms}–{affect.end_ms} ms (utterance scope)</dd>
    <dt>model</dt><dd><code>{html.escape(output.model.name)}</code></dd>
  </dl>
  <script id="attune-timeline" type="application/json">{data}</script>
  <script>
    const data = JSON.parse(document.getElementById("attune-timeline").textContent);
    const audio = document.querySelector("audio");
    const rails = Array.from(document.querySelectorAll(".rail"));
    rails.forEach((rail) => {{
      const head = document.createElement("div");
      head.className = "playhead";
      head.style.left = "0%";
      rail.appendChild(head);
      rail.addEventListener("click", (event) => {{
        if (!audio || !data.duration_ms) return;
        const box = rail.getBoundingClientRect();
        const ratio = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
        audio.currentTime = (ratio * data.duration_ms) / 1000;
      }});
    }});
    if (audio) {{
      const move = () => {{
        const ratio = data.duration_ms ? (audio.currentTime * 1000) / data.duration_ms : 0;
        document.querySelectorAll(".playhead").forEach((head) => {{
          head.style.left = (Math.min(1, Math.max(0, ratio)) * 100).toFixed(3) + "%";
        }});
      }};
      audio.addEventListener("timeupdate", move);
      audio.addEventListener("seeked", move);
    }}
  </script>
</body>
</html>
"""



def _affect_distribution_html(affect) -> str:
    """Project the schema category distribution; do not invent a label."""
    parts = []
    for label, value in affect.categories.items():
        name = label.value if hasattr(label, "value") else str(label)
        parts.append(f"{html.escape(name)} {float(value):.3f}")
    return "; ".join(parts)


def _banners(
    *,
    fixture: bool,
    dcase_head_configured: bool,
    frame_count: int,
    word_count: int,
) -> Iterable[str]:
    if fixture:
        yield ("Schema/CLI fixture. Not a model prediction and not DCASE localization.")
    if not dcase_head_configured:
        yield (
            "DCASE frame timestamps omitted: no gated DCASE checkpoint was configured. "
            "Full-width event/style bars are utterance scope, never localization. "
            "STARSS23 stays unwired."
        )
    elif frame_count == 0:
        yield (
            "DCASE gated head was configured but emitted no bounded laugh/cough/"
            "throat_clear spans on this clip. Remaining 0..duration bars are still "
            "utterance scope."
        )
    else:
        yield (
            "Solid bars are DCASE frame spans for laugh/cough/throat_clear that do "
            "not cover the whole clip. Hatched full-width bars remain utterance scope."
        )
    if word_count == 0:
        yield ("No genuine ASR word timestamps were returned. Attune does not interpolate words.")


def _span_view(
    channel: str,
    item: VocalEvent | VocalStyle,
    duration_ms: int,
) -> dict[str, object]:
    kind = classify_span(
        channel=channel,
        label=item.label.value,
        start_ms=item.start_ms,
        end_ms=item.end_ms,
        duration_ms=duration_ms,
    )
    return {
        "id": item.id,
        "channel": channel,
        "label": item.label.value,
        "start_ms": item.start_ms,
        "end_ms": item.end_ms,
        "confidence": item.confidence,
        "kind": kind,
    }


def _left_width(start_ms: int, end_ms: int, duration_ms: int) -> tuple[str, str]:
    if duration_ms <= 0:
        return "0%", "0%"
    left = 100.0 * start_ms / duration_ms
    width = 100.0 * max(end_ms - start_ms, 1) / duration_ms
    return f"{left:.3f}%", f"{width:.3f}%"


def _track_row(title: str, spans: list[dict[str, object]], duration_ms: int) -> str:
    heading = html.escape(title)
    if not spans:
        return (
            f'<section class="track"><h2>{heading}</h2>'
            f'<div class="rail"><div class="empty">none</div></div></section>'
        )
    blocks = []
    for span in spans:
        left, width = _left_width(int(span["start_ms"]), int(span["end_ms"]), duration_ms)
        label = html.escape(str(span["label"]))
        kind = html.escape(str(span["kind"]))
        title_attr = html.escape(
            f"{span['label']} {span['start_ms']}–{span['end_ms']} ms ({span['kind']})"
        )
        blocks.append(
            f'<div class="span {kind}" style="left:{left};width:{width}" title="{title_attr}">'
            f"{label} {span['start_ms']}–{span['end_ms']}ms</div>"
        )
    return (
        f'<section class="track"><h2>{heading}</h2>'
        f'<div class="rail">{"".join(blocks)}</div></section>'
    )


def _word_row(words: list[dict[str, object]], duration_ms: int) -> str:
    if not words:
        return (
            '<section class="track"><h2>Words</h2>'
            '<div class="rail"><div class="empty">no genuine word timestamps</div></div></section>'
        )
    blocks = []
    for word in words:
        left, width = _left_width(int(word["start_ms"]), int(word["end_ms"]), duration_ms)
        text = html.escape(str(word["text"]))
        title_attr = html.escape(f"{word['text']} {word['start_ms']}–{word['end_ms']} ms")
        blocks.append(
            f'<div class="span word" style="left:{left};width:{width}" '
            f'title="{title_attr}">{text}</div>'
        )
    return (
        f'<section class="track"><h2>Words</h2><div class="rail">{"".join(blocks)}</div></section>'
    )
