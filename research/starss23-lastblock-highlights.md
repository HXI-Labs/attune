# STARSS23 last-block encoder fine-tune (first-60s 100 ms highlights)

Status: protocol locked 2026-08-28 **before any train step**. This is **not**
second-listener Attune gold. Jerry authorized the orange gold-review 100 ms
overlays as good enough to train/eval against; do not claim independent
Attune gold.

## Labels

STARSS23 class-4 `laughter` 100 ms activity on the first 60 seconds of each
eligible development recording. Owner-authorized as good enough 2026-08-28
from the gold-review waveform highlights. Not second-listener gold. Not the
6 Jerry-retimed clips. Not a relabel.

Only STARSS23 class 4 maps to Attune `laugh`. Speech, footsteps, doors, and
all other classes stay unmapped.

## Split (locked)

Train/val come from official development rooms, first-60s only
(`source_window_start_ms == 0`). Inspection is the locked 48-event first-60s
pack (49 clips / 48 laughs / 29 true negatives; gold-review pack clip IDs).

- Train: first-60s development rooms **not** in `{sony-room21, tau-room6}`
  and **not** in the inspection set (file- and room-disjoint). Expected 43
  clips.
- Val / early-stop: first-60s `{sony-room21, tau-room6}` only. Expected 19
  clips. Inspection unused for fitting, early-stop, or decoder.
- Inspection: locked 49 first-60s clips / 48 events. Scored **once** after
  early-stop. Never used for fitting, early-stop, or decoder.

Do not train on later tiles. No tiling-in-train. Do not train on the locked 48-event inspection clips. Do not use only the 6 Jerry-retimed clips as the train set (N is too small). Do **not** claim the 6-clip listen
set is gold.

Reported frozen-MLP baseline remains **0.1395** (SHA family `0a27733`,
collar 0.1395 = 9/72/39, segment 0.5124 vs whole-clip 0.1721) until this run
beats it on inspection. Wire only at collar F1 ≥ 0.25.

## Architecture

Last SANM block `encoder.tp_encoders.19` unfrozen via
`LastBlockSenseVoiceFrameEncoder`. Frontend frozen (`dither=0`). Earlier
encoder blocks frozen (`encoders0`, `encoders`, `tp_encoders.0`–`.18`,
`after_norm`, `tp_norm`). Decoder (SenseVoice CTC/decoder) locked and unused.
`FrozenSenseVoiceEncoder` / `FrozenSenseVoiceFrameEncoder` freeze contracts
stay frozen (`requires_grad` remains false; no last-block leak into those
classes).

Two-layer laugh MLP `512 → 64 → 1`, initialized from the gitignored frozen
0.1395 head `artifacts/starss23-scene-raster/frame-head-40epoch.pt` if the
state dict is compatible, else random seed 0. No Conv/GRU. No decoder grid.

Frozen full-encoder embeddings under `artifacts/starss23-scene-raster/embeddings`
are **not** used: the last block changes. Prefix-cache frozen activations
into `tp_encoders.19` only.

## Decoder (locked to the 0.1395 / 0a27733 repaired recipe)

Copy of the JSON that produced inspection collar 0.1395. Do not search.

- high 0.95
- low 0.855
- gap 0 (`max_gap_frames`)
- min_active 1
- median 3
- onset shift 0

No decoder-search. No tiling. Do not lower the gate.

## Train recipe

- Script: `scripts/train_starss23_encoder_lastlayer.py`
- Device: CPU only
- Seed 0
- encoder-lr 5e-6 (closest DCASE last-block iterate)
- head-lr 1e-4
- Max 15 epochs, patience 5 on **val collar** F1
- Loss: unweighted frame BCE with logits (same family as the 0.1395 MLP;
  do not revive pos_weight / boundary-weighted BCE)
- One clip per step

## Predeclared gate

Write-once. Do not lower.

- Inspection collar F1 ≥ **0.25**
- Inspection segment F1 margin vs whole-clip ≥ **0.05**
- Inspection never used for fitting, early-stop, or decoder

If the gate fails, STARSS23 stays unwired. Record numbers. Do not merge.
`dcase_checkpoint_acceptance` continues to refuse STARSS23.

If the gate passes, infer may grow a STARSS23 laugh-only acceptance path that
still requires `dataset == starss23` and collar ≥ 0.25. Leave the PR open;
do not merge from this protocol.

## Licence

SenseVoiceSmall by FunASR/FunAudioLLM, FunASR Model Open Source License v1.1.
`ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1`. Fine-tuned last-block weights stay
private under `artifacts/starss23-lastblock-highlights/` (gitignored). Do not
commit audio, `.pt`, or `.venv`. Results JSON is numbers-only.

## Decision rule

JSON is authoritative. Tests must pass. Reported best remains 0.1395 until
inspection collar beats it; wiring still requires 0.25, not merely beating
0.1395.

## Result (CPU seed 0, after protocol lock)

Completed 10/15 epochs (patience 5 on validation collar). Last block
`encoder.tp_encoders.19` had 3,158,016 trainable parameters; MLP head 32,897;
total trainable 3,190,913. Head initialized from the frozen 0.1395 40-epoch
MLP. Prefix cache 215 s (111 first-60s clips); train ~76 s. Decoder locked to
0a27733 (high 0.95, low 0.855, gap 0, min_active 1, median 3). No decoder
grid. No tiling-in-train.

Inspection (locked 49 clips / 48 events; never used for fitting):

| metric | this run | gate | verdict |
|---|---:|---:|---|
| 200 ms collar F1 | 0.0282 (1/22/47) | ≥ 0.25 | fail |
| 1 s segment F1 | 0.4533 | — | — |
| whole-clip segment F1 | 0.1721 | — | — |
| segment margin vs whole-clip | +0.2812 | ≥ 0.05 | pass |
| val peak collar F1 | 0.3333 | early-stop only | — |

**STARSS23 stays unwired.** Collar 0.0282 does not beat reported best 0.1395
and does not clear 0.25. `dcase_checkpoint_acceptance` still refuses STARSS23.
Do not merge.
