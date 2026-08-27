# STARSS23 60-second scene raster timing

Machine-readable status is in `research/starss23-scene-raster-results.json`.
The scientific gold gate remains **closed**. Language is unverified. The
SenseVoiceSmall encoder stayed frozen. SenseVoiceSmall is by
FunASR/FunAudioLLM under the FunASR Model Open Source License Agreement v1.1.

## Protocol

The 10-second laughter-centered crop was the suspected collar killer. This
pass takes the **first 60 seconds** of every official STARSS23 v1.1
development recording that is at least 60 s and has no Music class 8 in that
excerpt. Labels stay at 100 ms. Only class 4 laughter maps to Attune `laugh`.
Overlapping laughter sources are unioned per frame; non-contiguous bursts stay
distinct. Official `dev-train` vs `dev-test` rooms and files remain disjoint.
Audio, embeddings, and checkpoints are gitignored.

Development yielded 62 scenes and inspection 49 scenes. Room-disjoint
validation used `sony-room21` and `tau-room6` (43/19). One frozen-frame MLP
(`sensevoice-small-encoder-frames-v1`, dither 0) and one seed were trained.
Hysteresis/min-duration decoding was selected on validation only. Inspection
was scored once.

## Held-out official test rooms

| Method | 1 s segment F1 | 200 ms collar event F1 |
|---|---:|---:|
| Frozen frame MLP | **0.4794** | **0.1074** |
| Whole-clip oracle tags | 0.1721 | 0.0000 |

Segment margin vs whole-clip is `+0.3073`, which clears the predeclared
`+0.05` requirement. Collar F1 is 8/48 matched intervals and **fails** the
fixed `0.25` requirement. STARSS23 laugh timestamps therefore remain
**unwired**. DCASE wiring is unchanged. Whole-clip `0..60000` tags are not
localization.

The 10-second crop MLP scored 0.7381 / 0.1159 collar. Replacing crops with
natural 60 s scenes did not lift collar F1 above 0.25. Natural-scene event
boundaries remain unsolved. No second architecture, seed, or DCASE rerun was
attempted.
