# DCASE last-block encoder fine-tune (isolated events)

Status: protocol locked before any train step. This run is **not** a 100 ms
STARSS23 label experiment. STARSS23 stays unwired. Jerry's 6-clip STARSS23
listen set is not used.

## Why this run exists

The frozen SenseVoiceSmall frame MLP on isolated DCASE 2016
laugh / cough / throat_clear is the current wired demo head:

| decoder | inspection collar F1 | source |
|---|---:|---|
| exact-match threshold 0.95 | 0.4637 | `research/dcase-frame-localization-results.json` |
| hysteresis (val-selected, now locked) | 0.5279 | same |
| segment F1 margin vs whole-clip | +0.3876 (≥ 0.05) | same |

Head SHA `cb74b1d493511d384dea3abf37139cf754a16f540b594b7f92f82647204b01fc`
at the local gitignored path
`/workspace/attune-dcase/artifacts/dcase-frame-localization/frame-head.pt`.

The STARSS23 100 ms path is dead (best collar 0.1395; listen-set median
|onset Δ| 392 ms; gate 0.25). It is not a fallback.

Hypothesis (non-binding; this run must verify, not assume): frozen ASR frames
cap isolated-event timing. Unfreezing only the last SenseVoice encoder block,
keeping the existing DCASE frame MLP head trainable, can beat 0.4637 exact
collar without wrecking lexical features on this small isolated set.

## Corpus

- Isolated DCASE 2016 Task 2 synthetic strong labels: laugh, cough, throat_clear.
- Manifests: `data/manifests/dcase2016-localization-training.jsonl` and
  `data/manifests/dcase2016-localization-inspection.jsonl`.
- Scene/file-disjoint development split from
  `scripts/train_temporal_localization.py` (168 train / 48 val / 100 inspection).
- 100 ms-irrelevant: DCASE events are real isolated intervals, not STARSS23
  100 ms grid labels.

## Architecture (inspected from local SenseVoiceSmall)

FunASR `SenseVoiceEncoderSmall` forward path:

1. `encoders0` (1 × `EncoderLayerSANM`)
2. `encoders` (49 × `EncoderLayerSANM`)
3. `after_norm`
4. `tp_encoders` (20 × `EncoderLayerSANM`)
5. `tp_norm`

The last encoder **block** is `encoder.tp_encoders.19` (~3.16M parameters).
That is the only unfrozen encoder module. Frontend stays frozen (`dither=0`).
`embed`, `encoders0`, `encoders`, earlier `tp_encoders`, `after_norm`,
`tp_norm`, CTC, and the decoder are frozen. `FrozenSenseVoiceEncoder` /
`FrozenSenseVoiceFrameEncoder` are unchanged: `_assert_frozen`,
`inference_mode`, and `detach` stay in force.

New class: `attune.models.sensevoice_last_block.LastBlockSenseVoiceFrameEncoder`.
Extraction route matches the frozen frame encoder (frontend dither 0, four
query frames stripped after the encoder) so frame hop (~60 ms) and first-frame
centre (~30 ms) stay comparable.

Final acoustic frames are **not** cached during training (last-block weights
change). Frozen **prefix** activations (input to `tp_encoders.19`) may be
cached because those weights do not change. Inspection eval may cache frames
only if the last block is frozen at eval time.

## Predeclared gate

Write-once. Do not lower. Do not grid-search the decoder.

- Inspection **exact-match** collar F1 must be ≥ 0.4637 (no regression vs the
  current wired head's exact decoder).
- Inspection **hysteresis** collar F1 must be ≥ 0.5279 to replace the wired
  demo head.
- Segment F1 margin vs whole-clip must stay ≥ 0.05.
- Decoder is locked to the current wired DCASE head / train script selection:
  exact threshold 0.95; hysteresis `high=0.95`, `low=0.855`, `max_gap_frames=1`,
  `min_active_frames=1`. No decoder search on this run.
- Seed 0. Last 1 encoder block only. MLP head trainable (initialized from the
  SHA-pinned wired head). Max 10 epochs, patience 3 on validation hysteresis
  collar F1, CPU.
- If the gate fails, keep the current frozen head wired. Do not touch STARSS23.
  Do not merge. Do not replace the PR 26 head.

## Train recipe

- Script: `scripts/train_dcase_encoder_lastlayer.py`
- Device: CPU (this machine reports `torch 2.13.0+cu130` / `cuda False`)
- Optimizer: AdamW; last-block LR 2e-5; head LR 1e-4; grad clip 5
- Loss: frame-wise BCE with logits, train-set positive weights from the same
  DCASE overlap targets as the frozen MLP
- One clip per step (SANM is sequence-local; frames are not flattened across
  clips)
- Epoch log written immediately so a crash still leaves validation collar
  numbers under `artifacts/dcase-encoder-lastlayer/` (gitignored weights) and
  numbers-only JSON under `research/dcase-encoder-lastlayer-results.json`

## Licence

SenseVoiceSmall by FunASR/FunAudioLLM, FunASR Model Open Source License v1.1,
reviewed 2026-08-26 (`data/provenance/sensevoice_small_weights.yaml`).
`ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1`. Fine-tuned last-block weights stay
private and gitignored. Do not commit audio, `.pt`, or `.venv`.

## Decision rule

JSON is authoritative. Tests must pass before any wiring. This branch must not
dirty PR 26 (`cursor/restore-dcase-isolated-demo`). Even if the gate clears,
do not merge from this protocol; replacement of the wired head is a separate
deliberate step.

## Result (CPU seed 0, after protocol lock)

Completed 9/10 epochs (patience 3 on validation hysteresis collar). Last block
`encoder.tp_encoders.19` had 3,158,016 trainable parameters; MLP head 66,051;
total trainable 3,224,067. Prefix cache 93 s; train ~51 s.

Inspection (locked decoder, never used for fitting):

| metric | this run | gate | verdict |
|---|---:|---:|---|
| exact-match collar F1 | 0.4755 | ≥ 0.4637 | hold (no regression) |
| hysteresis collar F1 | 0.5169 | ≥ 0.5279 | miss; do not replace wired head |
| segment margin vs whole-clip | +0.4059 | ≥ 0.05 | hold |

JSON: `research/dcase-encoder-lastlayer-results.json`. Weights remain gitignored.
The PR 26 frozen head stays wired. STARSS23 was not touched.

## Iterate 2 (predeclared 2026-08-28, before any retrain)

The seed-0 `encoder-lr 2e-5` / `head-lr 1e-4` run is frozen evidence: exact 0.4755 HOLD, hysteresis 0.5169 MISS vs 0.5279, segment margin +0.4059 HOLD. Val hysteresis peaked 0.6891 at epoch 6 then failed inspection transfer. Do **not** rerun that config. Do not lower the gate. Do not grid-search the decoder. Decoder stays locked (exact 0.95; hysteresis 0.95/0.855/gap 1/min 1). Last 1 encoder block remains `encoder.tp_encoders.19`. Head still initializes from the SHA-pinned PR 26 MLP (`cb74b1d4…`). Seed 0. CPU. Prefix cache reuse is allowed. STARSS23 stays unwired.

`--encoder-weight-decay` default is 0 so variants that omit it keep the original AdamW group construction (no extra last-block decay). Variant 3 is the only run that sets last-block-only decay.

Three variants, in this order. Stop early if any variant clears the full gate: inspection hysteresis collar ≥ 0.5279 AND exact ≥ 0.4637 AND segment margin ≥ 0.05. If one clears, the demo/infer path MAY point at that new checkpoint for isolated DCASE only; still do not merge; still do not touch STARSS23. If all three miss, last-block LR/WD is exhausted on this locked decoder and the PR 26 frozen head stays wired.

1. Lower LR: `--encoder-lr 5e-6 --head-lr 3e-5 --epochs 10 --patience 3`. JSON `research/dcase-encoder-lastlayer-lr5e6-results.json`. Checkpoint dir `artifacts/dcase-encoder-lastlayer-lr5e6/` (gitignored weights).
2. If 1 misses hysteresis: `--encoder-lr 5e-6 --head-lr 1e-4 --epochs 15 --patience 5`. JSON `research/dcase-encoder-lastlayer-lr5e6-head1e4-results.json`. Checkpoint dir `artifacts/dcase-encoder-lastlayer-lr5e6-head1e4/`.
3. If 2 misses: last-block-only weight decay `--encoder-lr 5e-6 --head-lr 3e-5 --encoder-weight-decay 0.05 --epochs 15 --patience 5`. JSON `research/dcase-encoder-lastlayer-wd-results.json`. Checkpoint dir `artifacts/dcase-encoder-lastlayer-wd/`.

## Iterate 2 result (CPU seed 0, after predeclaration)

All three predeclared variants missed hysteresis. Exact and segment margin held. Decoder was not searched. The PR 26 frozen head stays wired (`cb74b1d4…`). STARSS23 was not touched. Prefix cache reused (316 hits / 0 misses).

| variant | exact | hysteresis | margin | val peak | gate |
|---|---:|---:|---:|---:|---|
| 2.1 enc 5e-6 / head 3e-5 | 0.4910 | 0.5000 | +0.4070 | 0.6154 | keep |
| 2.2 enc 5e-6 / head 1e-4 | 0.4911 | 0.5247 | +0.4003 | 0.6897 | keep |
| 2.3 enc 5e-6 / head 3e-5 / wd 0.05 | 0.4910 | 0.5000 | +0.4070 | 0.6154 | keep |

Closest hysteresis was 0.5247 (variant 2.2) vs 0.5279. Last-block LR/WD is exhausted on this locked decoder. Table: `research/dcase-encoder-lastlayer-iterates.md`.
