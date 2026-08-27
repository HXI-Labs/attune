# Wav-in Attune cascade demo

Play one local WAV through the reviewed cascade and inspect **words** (only if
ASR returns genuine stamps), **affect**, and **events**. JSON is authoritative.
XML and the HTML timeline are deterministic projections of that JSON.

This is not a product UI. The scientific gold gate remains closed. STARSS23
timestamps stay unwired. A later natural-scene head must still clear collar F1
≥ 0.25 **and** segment margin ≥ 0.05 on the 48-event first-60s control; a new
representation cannot be another MLP, GRU, or Conv on frozen SenseVoice frames.

## What is timed, and what is not

- `laugh`, `cough`, and `throat_clear` receive DCASE **frame** spans only when a
  gated DCASE checkpoint is present, passes its loader, and emits bounds that
  are not `0..duration`.
- Every other event/style bar that covers the whole clip is **utterance scope**.
  It is not localization. The HTML hatches those bars so they cannot be mistaken
  for frame alignment.
- Missing ASR alignment stays `transcript.words = []`. Attune never interpolates
  word times.
- No STARSS23 / natural-scene audio. Demo WAVs must be isolated
  laugh/cough/throat_clear takes (DCASE-like or a recorded isolated clip).

If no DCASE checkpoint is configured, the cascade still runs: events remain, and
DCASE spans are omitted honestly. That run is **not** a timing demo.

## Local artifacts

Set `ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1` after the recorded SenseVoice licence
review. The CLI does not download weights. It will use explicit flags / env
vars, otherwise these gitignored local paths when present:

| Artifact | Local discovery |
|---|---|
| SenseVoice-Small | `ATTUNE_SENSEVOICE_SMALL_PATH`, `artifacts/starss23-scene-raster/sensevoice-small`, `data/raw/model-cache/sensevoice-small` |
| emotion2vec+ | `ATTUNE_EMOTION2VEC_PLUS_PATH`, `data/raw/model-cache/emotion2vec-plus` |
| VocalSound head | `ATTUNE_VOCALSOUND_PROBE_PATH`, `artifacts/event-probe/head.pt` |
| FSD50K head | `ATTUNE_FSD50K_PROBE_PATH`, `artifacts/fsd50k-event-probe/head.pt` |
| DCASE frame head (optional, one only) | `ATTUNE_TEMPORAL_HEAD_PATH`, `artifacts/dcase-frame-localization/frame-head.pt` |

A STARSS23 `frame-head*.pt` is never auto-selected. Passing one as
`--temporal-head` is an error.

```bash
export ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1
# Isolated laugh/cough/throat_clear WAV only. Never STARSS23 participant audio.
uv run python scripts/infer.py path/to/isolated.wav \
  --output research/demo/isolated.attune.json \
  --xml-output research/demo/isolated.attune.xml \
  --html-output research/demo/isolated.attune.html
```

Open the HTML file in a browser. Keep the WAV beside it (audio is gitignored).
`--fixture-mode` is a weight-free schema/CLI path and names itself a placeholder.

`display-fixture.*` in this directory is a schema-valid timeline sample used to
show frame-local vs utterance-scope bars. It is **not** cascade output.
