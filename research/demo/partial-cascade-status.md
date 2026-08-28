# Partial cascade inventory (2026-08-28)

Local DCASE-stamped wav demo restored. Gold gate remains closed. STARSS23
natural gold is paused. No Cloud Agents. No merge.

## DCASE gated frame-head

Restored locally. Gitignored checkpoint:

`artifacts/dcase-frame-localization/frame-head.pt`

sha256 `cb74b1d493511d384dea3abf37139cf754a16f540b594b7f92f82647204b01fc`

Retrain used the committed protocol: `scripts/prepare_dcase_localization.py`
then `scripts/train_temporal_localization.py` seed 0, 30 epochs, 66,051-param
`512→128→3` MLP on frozen `sensevoice-small-encoder-frames-v1`. Licence review
is `data/provenance/dcase2016_task2.yaml` (CC BY 3.0 train/dev, CC BY 4.0 public
test). Archive MD5s matched the provenance ledger. Derived-clip SHA-256 values
matched the committed 216/100 manifests.

Held-out inspection (never used for fitting or threshold selection) reproduced
the previously reported gated numbers exactly:

| Method | 1 s segment F1 | 200 ms collar F1 |
|---|---:|---:|
| Direct threshold | 0.7285 | 0.4637 |
| Validation-selected hysteresis | 0.7059 | 0.5279 |
| Whole-clip oracle tags | 0.3183 | 0.0000 |

Wiring gate margin is `+0.3876` (required `+0.05`). `scripts/infer.py` auto-
selects this DCASE head and refuses STARSS23 checkpoints.

## VocalSound frozen probe (utterance scope)

Restored locally. Gitignored checkpoint:

`artifacts/event-probe/head.pt`

sha256 `0dc451db80a72813abe5766ba17a0e4e3845e34df55475b11c36173af6588574`

Encoder frozen (`sensevoice-small-encoder-v2`). Head is
`linear_with_none_logit` (30,726 params). Train/val/test is 562/120/80 from
the same ten VocalSound Zenodo *validation* shards used by the inspection
cache — not a full archive dump. Sixteen inspection speakers were excluded
before the split. OOD none-logit negatives are CREMA-D (GitHub 16 kHz PCM as
published) plus locally converted FSD50K probe-pool clips. Committed FSD50K
conversion hashes were **not** rewritten.

Inspection-test macro-F1 is **0.8174** (laugh 0.8485, sigh 0.9091, cough
0.6154, throat_clear 0.8108, sneeze 0.9032). That beats committed stage-2
log-mel 0.4708 and off-the-shelf AED 0.3213. It does **not** beat the prior
full-pool SenseVoice frozen probe 0.8598
(`research/sensevoice-frozen-probe-metrics.json`) — different train pool.
`gate_passed` is **false**. The none-logit validation threshold is 0.9967
(OOD FPR 0). Metrics: `research/cascade-vocalsound-probe-metrics.json`.

Clip-level probe events are **utterance scope**, never frame timestamps.
When a gated DCASE span exists for the same label, the merge keeps the frame
span and drops the whole-clip probe bar.

## FSD50K frozen probe (closed-gate hole)

`artifacts/fsd50k-event-probe/head.pt` is absent. The committed protocol
report `research/fsd50k-frozen-probe-metrics.json` has inspection macro-F1
0.7410 and **`gate_decision: closed`**. Local ffmpeg rewrites of the 320
probe-pool clips do not match committed WAV SHA-256 values (container
headers). The 100 inspection-test clips were not prepared hash-faithfully
on this box. The gate is not lowered. Shout / whisper / sob / scream stay
omitted until a hash-faithful retrain can be run under the closed-gate
protocol. STARSS23 stays refused.

## STARSS23 heads (refused)

Present under `artifacts/starss23-scene-raster/` on some boxes and unwired.
`scripts/infer.py` refuses these as `--temporal-head` and never auto-selects them.

## What the wav CLI can actually run on this box

| Artifact | On disk |
|---|---|
| SenseVoice-Small | yes — `data/raw/model-cache/sensevoice-small/model.pt` sha256 `833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea` |
| emotion2vec+ | yes — `data/raw/model-cache/emotion2vec-plus/model.pt` sha256 `60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd` (revision `b318240bfe67db81a8c572ecb37ce9c3759b81c9`) |
| VocalSound `artifacts/event-probe/head.pt` | yes — sha256 `0dc451db80a72813abe5766ba17a0e4e3845e34df55475b11c36173af6588574` |
| FSD50K `artifacts/fsd50k-event-probe/head.pt` | no |
| DCASE frame-head | yes — `artifacts/dcase-frame-localization/frame-head.pt` |

`scripts/infer.py` therefore runs SenseVoice AED, gated DCASE frame spans for
laugh/cough/throat_clear, emotion2vec+ affect (temperature-scaled, validation
confidence threshold 0.8337), and the VocalSound probe (none-logit, utterance
scope). FSD50K remains omitted. STARSS23 stays refused. This is not a full
Phase 2 cascade.

## Live HTML that exists

`research/demo/isolated-dcase.attune.html` is a real cascade run on a gitignored
held-out DCASE 2016 Task 2 public-test 10 s window
(`dcase2016-inspection_test-test_1_ebr_0_nec_5_poly_1-090000`). It stamps
frame-local laugh, cough, and throat_clear. emotion2vec+ emits a calibrated
distribution (distress 0.677 peak) and abstains because that confidence is
below the validation threshold — this is not a missing-head skip. The
VocalSound probe is configured; this mixed isolated-event window adds no
hatched whole-clip bar (DCASE keeps the frame spans). Companion WAV is
gitignored. Screenshot: `research/demo/isolated-dcase-timeline.png`.

`research/demo/sensevoice-words.attune.html` is a real run on the official
SenseVoice English example (gitignored WAV). Fifteen genuine word stamps fire
and affect emits `neutral` at 0.959 (no abstain). The DCASE head is configured
and emits no bounded laugh/cough/throat_clear spans on this speech clip. The
VocalSound none-logit probe abstains (speech is OOD for this head).
Screenshot: `research/demo/sensevoice-words-timeline.png`.

`research/demo/vocalsound-sigh.attune.html` is a real run on inspection clip
`vocalsound-f0627_0_sigh` (speaker `f0627` was excluded from training;
gitignored WAV). The probe emits **sigh** as a hatched 0–2731 ms utterance-
scope bar (confidence 0.9995). That bar is not a timestamp. SenseVoice AED
also emits a whole-clip `breath`. The DCASE head emits a bounded
throat_clear 1140–1920 ms false positive on this sigh. ASR returns a genuine
60 ms `o` stamp — not interpolated. Affect abstains (surprise 0.715 < 0.8337).
Screenshot: `research/demo/vocalsound-sigh-timeline.png`.

`research/demo/sensevoice-only.attune.html` remains the earlier SenseVoice-only
projection (no DCASE head, affect missing-head abstain).
