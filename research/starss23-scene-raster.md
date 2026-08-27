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

## Protocol

The 10-second laughter-centered crop was the suspected collar killer. This
protocol takes the **first 60 seconds** of every official STARSS23 v1.1
development recording that is at least 60 s and has no Music class 8 in that
excerpt. Labels stay at 100 ms. Only class 4 laughter maps to Attune `laugh`.
Overlapping laughter sources are unioned per frame; non-contiguous bursts stay
distinct. Official `dev-train` vs `dev-test` rooms and files remain disjoint.
Audio, embeddings, and checkpoints are gitignored.

Development yielded 62 scenes and inspection 49 scenes. Room-disjoint
validation used `sony-room21` and `tau-room6` (43/19). One frozen-frame MLP
(`sensevoice-small-encoder-frames-v1`, dither 0) and one seed were trained.
Hysteresis decoding was selected on validation only. Inspection was scored
once per pass. No Conv/GRU/CRNN head was added. The encoder stayed frozen.

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

## Held-out official test rooms

| Pass | 1 s segment F1 | whole-clip segment F1 | 200 ms collar F1 | TP/FP/FN |
|---|---:|---:|---:|---:|
| Frozen frame MLP, 40 epochs, old decoder | 0.4794 | 0.1721 | 0.1074 | 8/93/40 |
| Frozen frame MLP, 73 epochs + 900 ms decoder | 0.3974 | 0.1721 | 0.0303 | 1/17/47 |
| Frozen frame MLP, 40 epochs + repaired decoder | 0.5124 | 0.1721 | 0.1395 | 9/72/39 |
| Frozen frame MLP, 40 epochs + onset-shift decoder | 0.3716 | 0.1721 | 0.0585 | 6/151/42 |
| Frozen frame MLP, 214-epoch boundary-weighted BCE | **0.3673** | 0.1721 | **0.0588** | 2/18/46 |
| Whole-clip oracle tags | 0.1721 | 0.1721 | 0.0000 | 0/20/48 |

This pass segment margin is `+0.1952` (clears `+0.05`). Collar F1 **fails**
`>= 0.25` and is worse than the 0.1395 decoder-validity pass, so the reported
cascade decoder/checkpoint remains that 40-epoch config. STARSS23 laugh
timestamps therefore remain **unwired**. DCASE wiring is unchanged.
Whole-clip `0..60000` tags are not localization.

Of 46 false negatives: (a) 12 overlap a prediction that fails the 200 ms
collar (median overlapping onset error 300 ms, MAE 366.7 ms; did **not**
improve vs the 0.1395 pass median 200 ms); (b) 29 are gold events the head
never fires; (c) 5 are decoder-suppressed. False positives are short
fragments (median 420 ms versus gold p25 600 ms). Stopped after this one
inspection eval. No new architecture. No further decoder grid.
