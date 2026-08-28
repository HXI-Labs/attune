# STARSS23 frozen-encoder first-60s MLP listen pack

Protocol written **before** training. Machine-readable status after the run is
`research/starss23-frozen-mlp-listenpack-results.json`. The listen-test pack is
`research/starss23-frozen-mlp-listenpack/`. SenseVoiceSmall is by
FunASR/FunAudioLLM under the FunASR Model Open Source License Agreement v1.1.
The encoder stays frozen. Gold stays closed. STARSS23 stays **unwired** in
product `infer` unless this inspection clears both gates below.

## Why this pass exists

Jerry authorized the STARSS23 **100 ms waveform highlights** as good enough
labels for a frozen-MLP retrain and a listen test. Those overlays are
`human_100ms_activity_not_attune_gold` (owner-authorized activity labels, not
second-listener Attune gold). This pass does **not** promote them to gold.

Reported best frozen first-60s MLP remains **collar F1 0.1395** (9/72/39) with
segment F1 0.5124 versus whole-clip 0.1721, from
`artifacts/starss23-scene-raster/frame-head-40epoch.pt` plus the repaired
decoder at commit `0a27733`. Tiled later-window trains and last-block encoder
FT (PR 29, collar 0.0282) are **out of scope**. This pass copies the first-60s
frozen MLP recipe, retrains on the 100 ms overlays, scores the locked
inspection pack once, and builds a local listen pack of orange 100 ms overlays
versus predicted spans.

## Gates (do not lower)

Wire STARSS23 into product infer only if **both**:

- inspection 200 ms collar F1 **≥ 0.25**
- 1 s segment F1 minus whole-clip segment F1 **≥ 0.05**

`wired: false` unless both clear. Do not search a decoder to sneak under the
bar. Do not replace 0.1395 as the reported best unless this inspection collar
strictly beats it **and** the wiring gate passes.

## Labels

- Train / val targets: STARSS23 class-4 laughter mapped to Attune `laugh`,
  rasterized onto frozen SenseVoice frames from the 100 ms overlays.
- Inspection gold: the same 100 ms overlays on the locked 49-clip / 48-event
  first-60s pack (`data/manifests/starss23-gold-review-pack.jsonl`,
  `source_window_start_ms == 0`). Not second-listener gold.
- Language unverified. Music class 8 skipped upstream. Truncated-at-60 s
  first-tile events stay in the 48-event control (matched to 0.1395).

## Split (inspection never for fit, early-stop, or decoder)

- Development first-60s only (`source_window_start_ms == 0`). **No tiling in
  train.** Later 60 s windows are not used.
- Train: development first-60s rooms except `sony-room21` and `tau-room6`
  (43 clips / 51 events).
- Val / early-stop: those two rooms, first-60s only (19 clips / 14 events).
- Inspection: gold-review pack, 49 clips / 48 events / 29 true negatives.
  File- and room-disjoint from development. Never used for mean/scale, loss,
  early-stop, or decoder.
- Whole files stay in one split.

## Model (copy the 0.1395 recipe; do not invent)

- Encoder: `FrozenSenseVoiceFrameEncoder` (SenseVoiceSmall). **0 trainable
  parameters.** CPU only. Optimizer is `AdamW(head.parameters(), lr=1e-3)`.
- Head: two-layer MLP `512→64→1` (32,897 params). Seed **0**. No Conv / GRU /
  attention / CRF. No last-block or last-two-block encoder FT.
- Audio: first 60 s, **mean of 4 tetrahedral MIC capsules**, 24 kHz → 16 kHz
  mono. Cache `data/raw/starss23-scene-raster` (hash-identical to tiled
  first-60s windows). Not max-RMS (`starss23-scene-raster-v2`).
- Embeddings: reuse `artifacts/starss23-scene-raster/embeddings`. Do not
  re-encode when audio SHA-256 matches the cache key. Refuse a silent
  re-extract (`cache_misses` must be 0 unless explicitly allowed).
- Loss: the 27df8e1 40-epoch trainer used `BCEWithLogitsLoss` with class
  `pos_weight = n_neg / n_pos` from **train frames only**. Later notes named
  that checkpoint `40_epoch_unweighted` to contrast the 73-epoch pos-weight
  *retry*; the 27df8e1 source that produced the 40-epoch weights used
  pos_weight. This pass copies that 27df8e1 loop: 40 epochs, patience 6, seed
  0. Early-stop on validation BCE. Inspection unused in the loss.
- Decoder **locked** to the 0a27733 cell that produced 0.1395. Copied, not
  searched:

  - high 0.95
  - low 0.855
  - gap 0
  - min_active 1 (60 ms)
  - median 3
  - onset shift 0

  No threshold grid, no hysteresis grid, no onset-shift grid. Validation
  sanity with this decoder does not select a decoder.

## Outputs

- Checkpoint (gitignored): `artifacts/starss23-frozen-mlp-listenpack/frame-head.pt`
  — must not overwrite `frame-head-40epoch.pt`. SHA-256 recorded in the results
  JSON.
- Results: `research/starss23-frozen-mlp-listenpack-results.json` with
  inspection collar TP/FP/FN, segment F1, whole-clip F1, margin, comparison to
  0.1395, `wired: false` unless the gate clears.
- Listen-test pack: 8–12 inspection clips (true laughs and true negatives).
  Orange = 100 ms overlay (not gold). Model predicted spans in a second color.
  Caption: STARSS23 is unwired in product infer; this pack is a research
  listen test. Local paths only. Do not public-tunnel. Wavs gitignored.
- Tests: inspection clips not in train; FrozenSenseVoice still 0 trainable;
  decoder not searched; gates not lowered.

## Out of scope

- Wiring STARSS23 if the gate fails.
- Last-block / last-two-block encoder fine-tunes (PR 29 already failed).
- Tiling later windows into train.
- Decoder search or lowering 0.25 / 0.05.
- Merging this PR. Cloud Agents. Cloning. Public tunnels.
