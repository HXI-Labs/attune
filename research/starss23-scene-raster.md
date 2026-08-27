# STARSS23 60-second scene raster timing

Machine-readable status is in `research/starss23-scene-raster-results.json`.
Duration diagnostics are in
`research/error-analysis/starss23-scene-raster-durations.json`.
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
once per pass. No Conv/GRU head was added.

## Held-out official test rooms

| Pass | 1 s segment F1 | whole-clip segment F1 | 200 ms collar F1 | TP/FP/FN |
|---|---:|---:|---:|---:|
| Frozen frame MLP, 40 epochs | **0.4794** | 0.1721 | **0.1074** | 8/93/40 |
| Frozen frame MLP, 73 epochs + gold-percentile decoder | 0.3974 | 0.1721 | 0.0303 | 1/17/47 |
| Whole-clip oracle tags | 0.1721 | 0.1721 | 0.0000 | 0/20/48 |

The 40-epoch segment margin is `+0.3073`. The longer pass is `+0.2253`. Both
clear the predeclared `+0.05` requirement. Both **fail** collar F1 `>= 0.25`.
STARSS23 laugh timestamps therefore remain **unwired**. DCASE wiring is
unchanged. Whole-clip `0..60000` tags are not localization.

The longer pass kept the same 32,897-parameter MLP. Class weight came from
TRAIN frames only. Decoder min-duration candidates were train-gold length
percentiles (3/8/15 frames). Gap-merge searched `{0,2,4,8,12}` and a median
filter `{1,3,5}`. Validation selected high `0.95`, low `0.855`, gap `0`,
min-active `15` (900 ms), median window `5`. That recipe cut false positives
93→17, but those FPs were **not** short fragments (median 1080 ms versus gold
p25 600 ms; 0/17 shorter than gold p25). True positives fell 8→1 and misses
rose 40→47. Remaining error is misses, not fragments.

The 10-second crop MLP scored 0.7381 / 0.1159 collar. Replacing crops with
natural 60 s scenes, then training longer with a more conservative decoder,
did not lift collar F1 above 0.25. Natural-scene event boundaries remain
unsolved. No second architecture, seed, or DCASE rerun was attempted.
