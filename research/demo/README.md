# Wav-in Attune cascade demo

Play one local WAV through whatever reviewed cascade artifacts are on disk and
inspect **words** (only if ASR returns genuine stamps), **affect**, and
**events**. JSON is authoritative. XML and the HTML timeline are deterministic
projections of that JSON.

This is not a product UI. The scientific gold gate remains closed. STARSS23
timestamps stay unwired. A later natural-scene head must still clear collar F1
≥ 0.25 **and** segment margin ≥ 0.05 on the 48-event first-60s control; a new
representation cannot be another MLP, GRU, or Conv on frozen SenseVoice frames.

## Live DCASE-stamped HTML

`isolated-dcase.attune.json` / `.xml` / `.html` is a real `scripts/infer.py` run
with the locally retrained gated DCASE frame-head plus emotion2vec+ and the
VocalSound frozen probe. The gitignored WAV is a held-out DCASE 2016 Task 2
public-test 10 s window
(`dcase2016-inspection_test-test_1_ebr_0_nec_5_poly_1-090000`) containing
laugh, cough, and throat_clear. Spans are model output, not interpolated.

The head was reproduced with the committed protocol (seed 0, 30 epochs). Direct
threshold 0.7285 / 0.4637 and hysteresis 0.7059 / 0.5279 on the untouched
100-clip inspection matched the prior gated result. STARSS23 heads stay refused.

Affect is a real emotion2vec+ distribution. On this isolated-event clip it
abstains after calibration (distress 0.677 < 0.8337 threshold), not because
the head is missing. The VocalSound probe is configured; this mixed window
does not add a hatched whole-clip bar. FSD50K remains omitted. Inventory:
`research/demo/partial-cascade-status.md`. Screenshot:
`research/demo/isolated-dcase-timeline.png`.

Regenerate with:

```bash
export ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1
export ATTUNE_TEMPORAL_HEAD_PATH=artifacts/dcase-frame-localization/frame-head.pt
uv run python scripts/infer.py research/demo/isolated-dcase.wav \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --emotion2vec-path data/raw/model-cache/emotion2vec-plus \
  --vocalsound-probe artifacts/event-probe/head.pt \
  --temporal-head artifacts/dcase-frame-localization/frame-head.pt \
  --output research/demo/isolated-dcase.attune.json \
  --xml-output research/demo/isolated-dcase.attune.xml \
  --html-output research/demo/isolated-dcase.attune.html
```

## Speech-example HTML (words + emitted affect)

`sensevoice-words.attune.json` / `.xml` / `.html` is a real cascade run on the
official SenseVoice English example. Fifteen genuine word timestamps fire.
emotion2vec+ emits `neutral` at 0.9593 (no abstain). The DCASE head is present
and finds no bounded laugh/cough/throat_clear on this clip. The VocalSound
none-logit probe abstains (speech is OOD for this head). This is not a timing
demo for events. Screenshot: `research/demo/sensevoice-words-timeline.png`.

## VocalSound sigh HTML (clip-level probe, not a timestamp)

`vocalsound-sigh.attune.json` / `.xml` / `.html` is a real cascade run on
inspection clip `vocalsound-f0627_0_sigh` (speaker `f0627` excluded from
training; gitignored WAV). The frozen VocalSound probe emits **sigh** as a
hatched 0–2731 ms utterance-scope bar (confidence 0.9995). That bar covers
the whole clip on purpose: it is **not** localization. SenseVoice AED also
emits a whole-clip `breath`. The DCASE head emits a bounded throat_clear
1140–1920 ms false positive. ASR returns a genuine 60 ms `o` stamp. Affect
abstains (surprise 0.715 < 0.8337). Screenshot:
`research/demo/vocalsound-sigh-timeline.png`.

Regenerate with:

```bash
export ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1
uv run python scripts/infer.py research/demo/vocalsound-sigh.wav \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --emotion2vec-path data/raw/model-cache/emotion2vec-plus \
  --vocalsound-probe artifacts/event-probe/head.pt \
  --temporal-head artifacts/dcase-frame-localization/frame-head.pt \
  --output research/demo/vocalsound-sigh.attune.json \
  --xml-output research/demo/vocalsound-sigh.attune.xml \
  --html-output research/demo/vocalsound-sigh.attune.html
```

## What is timed, and what is not

- `laugh`, `cough`, and `throat_clear` receive DCASE **frame** spans only when a
  gated DCASE checkpoint is present, passes its loader, and emits bounds that
  are not `0..duration`.
- Every other event/style bar that covers the whole clip is **utterance scope**.
  It is not localization. The HTML hatches those bars so they cannot be mistaken
  for frame alignment. VocalSound probe labels (including sigh/sneeze) stay in
  this bucket even when they fire.
- Missing ASR alignment stays `transcript.words = []`. Attune never interpolates
  word times.
- No STARSS23 / natural-scene audio. Demo WAVs must be isolated
  laugh/cough/throat_clear takes (DCASE-like or a recorded isolated clip) or
  licence-clean inspection VocalSound / SenseVoice examples.

If no DCASE checkpoint is configured, SenseVoice AED events still run when
SenseVoice is local, and DCASE spans are omitted honestly. That run is **not**
a timing demo.

## Local artifacts

Set `ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1` after the recorded SenseVoice licence
review. The CLI does not download weights. SenseVoice-Small is required.
emotion2vec+, VocalSound, FSD50K, and the DCASE frame-head are used when present
and omitted with an HTML banner when absent. Explicit `--*-path` values that do
not exist remain hard errors. Evaluation (`scripts/evaluate_attune_cascade.py`)
still requires the full Phase 2 package.

| Artifact | Local discovery |
|---|---|
| SenseVoice-Small (required) | `ATTUNE_SENSEVOICE_SMALL_PATH`, `artifacts/starss23-scene-raster/sensevoice-small`, `data/raw/model-cache/sensevoice-small` |
| emotion2vec+ (optional) | `ATTUNE_EMOTION2VEC_PLUS_PATH`, `data/raw/model-cache/emotion2vec-plus` |
| VocalSound head (optional) | `ATTUNE_VOCALSOUND_PROBE_PATH`, `artifacts/event-probe/head.pt` |
| FSD50K head (optional) | `ATTUNE_FSD50K_PROBE_PATH`, `artifacts/fsd50k-event-probe/head.pt` |
| DCASE frame head (optional, one only) | `ATTUNE_TEMPORAL_HEAD_PATH`, `artifacts/dcase-frame-localization/frame-head.pt` |

A STARSS23 `frame-head*.pt` is never auto-selected. Passing one as
`--temporal-head` is an error.

On this box the FSD50K probe is absent and its inspection gate remains
**closed** (`research/fsd50k-frozen-probe-metrics.json`). Shout/whisper/sob/
scream are omitted rather than lowering that gate.

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


## SenseVoice-only live projection on this box

`sensevoice-only.attune.json` / `.xml` / `.html` is a real SenseVoice-Small run
on the official SenseVoice English example (converted locally to 16 kHz mono
WAV, gitignored as `research/demo/sensevoice-example-en.wav`). Affect abstains
because emotion2vec+ is absent. There are no laugh/cough/throat_clear events and
no DCASE frame spans. This is **not** a timing demo. Regenerate with:

```bash
export ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1
uv run python scripts/infer.py research/demo/sensevoice-example-en.wav \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --output research/demo/sensevoice-only.attune.json \
  --xml-output research/demo/sensevoice-only.attune.xml \
  --html-output research/demo/sensevoice-only.attune.html
```
