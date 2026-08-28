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

## STARSS23 heads (refused)

Present under `artifacts/starss23-scene-raster/` on some boxes and unwired.
`scripts/infer.py` refuses these as `--temporal-head` and never auto-selects them.

## What the wav CLI can actually run on this box

| Artifact | On disk |
|---|---|
| SenseVoice-Small | yes — `data/raw/model-cache/sensevoice-small/model.pt` sha256 `833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea` |
| emotion2vec+ | no |
| VocalSound `artifacts/event-probe/head.pt` | no |
| FSD50K `artifacts/fsd50k-event-probe/head.pt` | no |
| DCASE frame-head | yes — `artifacts/dcase-frame-localization/frame-head.pt` |

`scripts/infer.py` therefore runs SenseVoice AED plus gated DCASE frame spans
for laugh/cough/throat_clear, and abstains affect. This is not a full Phase 2
cascade. Inspection evaluation still needs the complete probe package.

## Live HTML that exists

`research/demo/isolated-dcase.attune.html` is a real cascade run on a gitignored
held-out DCASE 2016 Task 2 public-test 10 s window
(`dcase2016-inspection_test-test_1_ebr_0_nec_5_poly_1-090000`). It stamps
frame-local laugh, cough, and throat_clear. Affect abstains. Companion WAV is
gitignored.

`research/demo/sensevoice-only.attune.html` remains the earlier SenseVoice-only
projection (no DCASE head).
