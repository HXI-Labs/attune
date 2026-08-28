# STARSS23 60-second scene raster timing

Machine-readable status is in `research/starss23-scene-raster-results.json`.
Duration diagnostics are in
`research/error-analysis/starss23-scene-raster-durations.json`.
The validation decoder ablation is in
`research/error-analysis/starss23-decoder-ablation.json`.
The onset-shift search is in
`research/error-analysis/starss23-onset-shift-ablation.json`.
The scientific gold gate remains **closed**. Language is unverified. The
SenseVoiceSmall encoder stayed frozen. SenseVoiceSmall is by
FunASR/FunAudioLLM under the FunASR Model Open Source License Agreement v1.1.

**This pass is the Kyoto first-60s-val mean-of-4 tiled MLP, and it is
negative.** Train is 111 tiled windows from non-val rooms (**0** val-room
later tiles in train). Early-stop is **only** the 19 first-60s clips in
`sony-room21` / `tau-room6`. Later tiles of those rooms stay unused (same
recording would leak if they entered train; they are not extra train N).
Tiled inspection collar F1 is **0.0296** (2/25/106 on 108 gold). First-60s
control is 0.03125 (1/15/47) versus prior best **0.1395**. Segment margin
still clears +0.05; collar fails >=0.25. Gate **FAIL**. STARSS23 stays
**unwired**. **0.1395 remains the reported best and must not be replaced.**
Decoder locked to 0a27733; a miss cannot be blamed on tiling vs decoder
mismatch. Stopped after this one tiled inspection eval. Scored audio is
mean-of-4 in `data/raw/starss23-scene-raster-tiled`.
`data/raw/starss23-scene-raster-v2` is **max-RMS audio only** (259 files)
and was **not** this eval; do not mix it into tiled embeddings. Checkpoint
wrote to `artifacts/starss23-scene-raster/frame-head-tiled-valfirst60s.pt`
and did not touch `frame-head-40epoch.pt`.

## Protocol

This pass replaces first-60s-only scenes with **non-overlapping 60 s tiles**.
A tile is kept only when **both** WAV duration and CSV annotated extent cover
the full 60 s. Music class 8 is skipped. Unlabeled WAV tails past the
annotation and remainders under 60 s are not padded or tiled. Clip IDs include
`window_start_ms` (`…-w000000`, `…-w060000`, …). Downmix is the **mean of the
four unlabeled tetrahedral MIC capsules** over the whole window (omni, not
FOA W, not max-RMS, no in-window channel switch), then 24 kHz → 16 kHz mono.
New audio cache: `data/raw/starss23-scene-raster-tiled`. New embedding cache:
`artifacts/starss23-scene-raster/embeddings-tiled`. First-60s embeddings were
not reused. `data/raw/starss23-scene-raster-v2` is a leftover **max-RMS**
wav cache (259 files) and is **not** the scored protocol; do not point
prepare/train/embeddings at v2.

Laugh events that span a 60 s cut are dropped from train targets and from
scored gold on **later tiles only** (`window_start_ms > 0`). On the first tile
(`window_start_ms == 0`) truncated-at-60 s events are **kept**, so the
first-60s inspection subset remains the matched **48-event** control. Whole
files stay in one split. Validation rooms stay `sony-room21` and `tau-room6`.
Tiles from one file are correlated, not extra i.i.d. N.

Development: **150** tiles / **195** events (**11** later-tile spanning drops).
Inspection: **109** tiles / **108** events (**7** spanning drops). Room-disjoint
validation rooms are `sony-room21` and `tau-room6`. **Kyoto split:** train
keeps every tiled window except any window from those two rooms (**111**
clips / **175** events). Early-stop / val loss uses **only** their 19
first-60s clips (`window_start_ms == 0`, 14 events). Their later tiles
(**20** clips / 6 events) stay unused. First-60s subset of inspection
remains 49 clips / 48 events (three truncated-at-60 s events kept).

This pass trains the same two-layer MLP (`512→64→1`, **32,897** params), seed
0, unweighted BCE, early-stop on first-60s val BCE patience 25, cap 400.
Encoder frozen. Decoder **locked** to 0a27733: high 0.95, low 0.855, gap 0,
min-active 1, median 3, onset shift 0. **No decoder grid.** Dual eval with
that locked decoder: (A) first-60s inspection subset, gold 48, vs 0.1395
control (9/72/39); (B) tiled inspection 108 events = wiring gate. A miss
cannot be blamed on tiling vs decoder mismatch.

## Kyoto first-60s-val tiled MLP pass

Training stopped at **226** epochs (best first-60s val BCE 0.049434 near
epoch 201). Validation sanity with the fixed decoder was collar 0.2353
(2/1/12 on 19 clips); that score did not select a decoder. Embedding cache
hits 239 / misses 0.

### Tiled inspection (wiring gate; 109 clips / 108 events)

| Method | 1 s segment F1 | whole-clip segment F1 | 200 ms collar F1 | TP/FP/FN |
|---|---:|---:|---:|---:|
| Frozen frame MLP, 226-epoch Kyoto first-60s val | 0.3143 | 0.1538 | **0.0296** | 2/25/106 |
| Whole-clip oracle tags | 0.1538 | 0.1538 | 0.0000 | — |

Segment margin is `+0.1604` (**passes** `+0.05`). Collar F1 **fails** `>=0.25`.
Of 106 false negatives: (a) 21 overlap a prediction that fails the 200 ms
collar (median overlapping onset error 420 ms, MAE 412 ms; did **not**
improve vs the first-60s 0.1395 pass median 200 ms); (b) 83 are gold events
the head never fires; (c) 2 are decoder-suppressed. STARSS23 laugh timestamps
remain **unwired**. DCASE wiring is unchanged. Stopped after this one tiled
inspection eval.

### First-60s subset (matched 48-event control)

| Method | 1 s segment F1 | whole-clip segment F1 | 200 ms collar F1 | TP/FP/FN |
|---|---:|---:|---:|---:|
| This pass, `window_start_ms == 0` only | 0.3623 | 0.1721 | 0.03125 | 1/15/47 |
| Prior best first-60s (40-epoch + repaired decoder) | 0.5124 | 0.1721 | **0.1395** | 9/72/39 |
| Whole-clip oracle tags | 0.1721 | 0.1721 | 0.0000 | 0/20/48 |

Same locked decoder. First-60s gold is 49 clips / 48 events / 0 spanning
drops, including three events truncated at t=60 s. Collar 0.03125 is worse
than 0.1395 on that control, so 0.1395 stays the reported best. Early-stopping
on first-60s val instead of all 39 val-room tiles did not recover the 0.1395
control. This subset is **not** the wiring gate.

## Prior tiled mean-of-4 MLP pass (full tiled val)

Training stopped at **217** epochs (best val BCE 0.043739 near epoch 192).
Validation sanity with the fixed decoder was collar 0.1739 (2/1/18); that
score did not select a decoder.

### Tiled inspection (wiring gate; 109 clips / 108 events)

| Method | 1 s segment F1 | whole-clip segment F1 | 200 ms collar F1 | TP/FP/FN |
|---|---:|---:|---:|---:|
| Frozen frame MLP, 217-epoch tiled mean-of-4 | 0.3143 | 0.1538 | **0.0148** | 1/26/107 |
| Whole-clip oracle tags | 0.1538 | 0.1538 | 0.0000 | — |

Segment margin is `+0.1604` (**passes** `+0.05`). Collar F1 **fails** `>=0.25`.
Of 107 false negatives: (a) 22 overlap a prediction that fails the 200 ms
collar (median overlapping onset error 380 ms, MAE 390 ms; did **not**
improve vs the first-60s 0.1395 pass median 200 ms); (b) 83 are gold events
the head never fires; (c) 2 are decoder-suppressed. STARSS23 laugh timestamps
remain **unwired**. DCASE wiring is unchanged. Stopped after this one tiled
inspection eval.

### First-60s subset (matched 48-event control)

| Method | 1 s segment F1 | whole-clip segment F1 | 200 ms collar F1 | TP/FP/FN |
|---|---:|---:|---:|---:|
| This pass, `window_start_ms == 0` only | 0.3741 | 0.1721 | 0.0308 | 1/16/47 |
| Prior best first-60s (40-epoch + repaired decoder) | 0.5124 | 0.1721 | **0.1395** | 9/72/39 |
| Whole-clip oracle tags | 0.1721 | 0.1721 | 0.0000 | 0/20/48 |

Same locked decoder. First-60s gold is 49 clips / 48 events / 0 spanning
drops, including three events truncated at t=60 s. Collar 0.0308 is worse
than 0.1395 on that control; tiling later windows into train did not help
the matched first-60s set. This subset is **not** the wiring gate.

## Decoder-validity pass

The previous 73-epoch pos-weight run selected min-active 15 (900 ms) because
the grid used train-gold p50 and broke ties by minimizing FP. Gold inspection
events have min 300 ms and p25 600 ms, so a 900 ms floor cannot recall short
gold. That pass did **not** retrain. It repaired the decoder search on
validation rooms only, then scored inspection once.

Repaired grid: min-active `{1, 2, 3}` plus train-gold p10, capped at p25
(8 frames); p50 (15 frames) excluded; gaps `{0, 2, 4, 8}`; median `{1, 3}`;
thresholds unchanged. Selection key is `(collar F1, recall, segment F1)`.

### Validation-only ablation (inspection unused)

| Cell | collar F1 | recall | segment F1 | TP/FP/FN |
|---|---:|---:|---:|---:|
| 40-epoch + old decoder | 0.0769 | 0.143 | 0.3133 | 2/36/12 |
| 40-epoch + repaired search | **0.1000** | 0.143 | 0.2899 | 2/24/12 |
| 73-epoch pos-weight + old decoder | 0.0333 | 0.071 | 0.2889 | 1/45/13 |
| 73-epoch pos-weight + repaired search | 0.0465 | 0.071 | 0.3099 | 1/28/13 |

The 40-epoch checkpoint won on validation collar F1 under the repaired
search (0.1000 vs 0.0465). Selected decoder: high `0.95`, low `0.855`, gap
`0`, min-active `1` (60 ms), median window `3`.

## Onset-shift pass

The repaired decoder's inspection misses were mostly predicted events that
failed the 200 ms collar (median overlapping onset error 200 ms). This pass
did **not** retrain and did **not** invent a new head. It searched a
validation-only onset/hysteresis repair of the 40-epoch checkpoint:

- median locked to `{1}` (no 3/5 smear)
- high thresholds unchanged
- `low_ratio` `{0.2, 0.3, 0.5, 0.7}`; `0.9` banned
- gaps `{0, 2, 4, 8}`
- min-active `{1, 2, 3}` plus train-gold p10, capped at p25; p50 unused
- global onset shift `{-180, -120, -60, 0}` ms on predicted `start_ms`,
  selected on validation rooms only

Selection key remains `(collar F1, recall, segment F1)`. Do not break ties
by minimizing FP. 1152 cells. Inspection unused until one winner eval.

### Validation-only onset-shift search (inspection unused)

| Onset shift | collar F1 | recall | segment F1 | TP/FP/FN |
|---|---:|---:|---:|---:|
| -180 ms | 0.0215 | 0.071 | 0.2111 | 1/78/13 |
| **-120 ms** | **0.0215** | **0.071** | **0.2147** | **1/78/13** |
| -60 ms | 0.0215 | 0.071 | 0.2081 | 1/78/13 |
| 0 ms | 0.0215 | 0.071 | 0.2130 | 1/78/13 |

Winner: high `0.9`, low `0.63`, gap `8`, min-active `1` (60 ms), median `1`,
onset shift `-120` ms. Collar F1 and recall tied across shifts; `-120` wins
on segment F1. Banning `low_ratio` 0.9 and locking median to 1 dropped
validation collar F1 from 0.1000 to 0.0215.

## Boundary-weighted BCE pass

Hypothesis: 200 ms collar misses are late onsets from uniform BCE, not a
global clock. One retrain of the same two-layer MLP, frozen encoder, seed 0.
The one change is onset/boundary-weighted BCE on TRAIN frames only: first and
last active gold frames get weight 5, ±1 neighbours get weight 3, interior
and background stay 1. No 44× class `pos_weight`. Early-stop on unweighted
validation BCE, patience 25, cap 400 epochs. Inspection decoding used the
predeclared 0a27733 decoder (high 0.95, low 0.855, gap 0, min-active 1,
median 3, onset shift 0). No decoder grid. Inspection unused for weights.

Training stopped at **214** epochs. Validation sanity with the fixed decoder
was collar 0.25 (2/0/12); that score did not select a decoder.

## BiGRU pass

Hypothesis: a small recurrent head can use temporal context the per-frame
MLP cannot. One predeclared architecture, one train, one inspection. Head:
1-layer bidirectional GRU, hidden 64 (128 concat), dropout 0, then
Linear(128→1). Train per-clip sequences with padding/mask so padded frames
do not enter the loss. Seed 0, AdamW 1e-3, unweighted BCEWithLogits (no 44×
`pos_weight`, no boundary weights). Early-stop on masked validation loss,
patience 25, cap 400. Encoder frozen. Decode with the predeclared 0a27733
decoder only. No decoder grid.

Training stopped at **86** epochs (best masked val at 61). Trainable params
**222,081**. Validation sanity with the fixed decoder was collar 0.25
(2/0/12); that score did not select a decoder.

## Held-out official test rooms (first-60s gold, 48 events)

These rows share the first-60s 49-clip / 48-event inspection set. The tiled
pass is **not** in this table; see dual eval above.

| Pass | 1 s segment F1 | whole-clip segment F1 | 200 ms collar F1 | TP/FP/FN |
|---|---:|---:|---:|---:|
| Frozen frame MLP, 40 epochs, old decoder | 0.4794 | 0.1721 | 0.1074 | 8/93/40 |
| Frozen frame MLP, 73 epochs + 900 ms decoder | 0.3974 | 0.1721 | 0.0303 | 1/17/47 |
| Frozen frame MLP, 40 epochs + repaired decoder | 0.5124 | 0.1721 | 0.1395 | 9/72/39 |
| Frozen frame MLP, 40 epochs + onset-shift decoder | 0.3716 | 0.1721 | 0.0585 | 6/151/42 |
| Frozen frame MLP, 214-epoch boundary-weighted BCE | 0.3673 | 0.1721 | 0.0588 | 2/18/46 |
| Frozen frame BiGRU, 86-epoch masked BCE | 0.2121 | 0.1721 | 0.0339 | 1/10/47 |
| Prior tiled MLP first-60s subset (full tiled val) | 0.3741 | 0.1721 | 0.0308 | 1/16/47 |
| This pass first-60s subset (Kyoto first-60s val) | 0.3623 | 0.1721 | 0.03125 | 1/15/47 |
| Whole-clip oracle tags | 0.1721 | 0.1721 | 0.0000 | 0/20/48 |

Wiring uses tiled inspection only. Collar there is 0.0296 < 0.25, so STARSS23
laugh timestamps remain **unwired**. This Kyoto first-60s-val mean-4 tiled MLP
is a **negative** result and **must not replace 0.1395**. A miss cannot be
blamed on tiling vs decoder mismatch (decoder locked to 0a27733). DCASE
wiring is unchanged. Whole-clip `0..60000` tags are not localization. Gold
remains closed. Language unverified.
