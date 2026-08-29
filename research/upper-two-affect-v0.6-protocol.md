# Upper-two affect candidate v0.6 protocol

Status: predeclared after the frozen v0.6 evaluation and before adapted
training.

## Reason for adaptation

The frozen full-head candidate did not pass its declared affect or robustness
gates:

| Evaluation | Required | Observed |
|---|---:|---:|
| Development affect macro-F1 | 0.60 | 0.594 |
| Speaker-disjoint SUBESCO affect macro-F1 | 0.60 | 0.537 |
| External RAVDESS affect macro-F1 | 0.40 | 0.327 |
| British speech-control auxiliary false-positive rate | 0 | 0.02 after development-only calibration |

The model did not collapse to one affect class. Its largest development and
sealed prediction shares were 0.208 and 0.279. Selective prediction reduced
error, and Acoustic Preference Score was +0.400 on development and +0.518 on
sealed conflict clips. The main affect errors were speaker- and corpus-specific:
SUBESCO joy was frequently mapped to `other` or `surprise`, while external
RAVDESS was over-assigned to `other`. Event analysis also showed that
`throat_clear` did not separate ordinary speech reliably across accents.

These results support adapting acoustic representations rather than widening
the existing heads or changing the label ontology.

## Candidate

- Train from `artifacts/manifests/joint-affect-v0.7.jsonl` (SHA-256
  `a7d4e13aaa162353409a5ce23e65b19fa8ee10fd89b18109852757999c25e238`).
  It retains all 1,458 development and 2,161 regression-test rows while capping
  training clips at 10 seconds. This removes 101 of 8,266 training clips whose
  quadratic attention cost dominated local runtime; evaluation still covers
  clips through 30 seconds.
- Warm start the Attune heads from
  `artifacts/training/local-frozen-full-head-v0.6/model.pt`.
- Adapt only the upper two SenseVoice encoder blocks and their final norms.
- Keep the lower encoder, CTC projection, and a copied base ASR tail frozen.
- Route ASR through the frozen tail and perception through the adapted tail.
- Exclude CTC loss because the exported ASR route is isolated from every
  trainable parameter.
- Train an initial six epochs with patience 3. Resume to the original maximum
  of 10 only if the sixth validation loss is still improving and held-out
  evaluation justifies the additional run.
- Use corpus-balanced, duration-bucketed batches and seed 42.
- Use physical batches of 6 with four-step accumulation for an effective batch
  size of 24 on Apple MPS. The first attempt needlessly evaluated the frozen
  ASR copy even though CTC output was disabled and exceeded the 9.07 GiB MPS
  limit before the first progress interval. The training-only forward now skips
  that unused branch. A full forward/backward pass over the 12 longest training
  clips (23.683 to 29.374 seconds) completed with all 60 expected gradient
  tensors before the restart. Export and normal inference still execute the
  frozen ASR route. A subsequent batch of 12 passed that one-step stress test
  but exceeded the memory limit after AdamW allocated its moment buffers. It
  also stopped before an epoch or checkpoint was written. A batch of 10 still
  reached the limit during repeated training steps. Batch 6 is the largest size
  observed to pass a 20-step training interval with optimizer state allocated;
  it leaves headroom rather than disabling MPS safeguards.
- Use learning rates of 2e-5 for existing heads and 5e-6 for adapted encoder
  parameters.

Authoritative configuration:
`configs/training/local-mps-upper-two-affect-v0.6.json`.

## Command

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/train_joint.py \
  --manifest artifacts/manifests/joint-affect-v0.7.jsonl \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --output-dir artifacts/training/local-upper-two-affect-v0.6 \
  --adaptation-policy upper_two \
  --preserve-base-asr \
  --initial-checkpoint artifacts/training/local-frozen-full-head-v0.6/model.pt \
  --config configs/training/local-mps-upper-two-affect-v0.6.json
```

## Acceptance

The adapted candidate uses the same gates as the frozen protocol. In addition:

- ASR logits must match a fresh base-model export before quantization.
- No localized event label remains enabled if it cannot achieve zero false
  positives on development speech controls.
- The already-inspected British controls are now development evidence. A new,
  untouched British control sample is required before final release.
- The original hostile-delivery recording must be retested before release.

RAVDESS and the original sealed split have now informed architecture selection,
so they remain regression sets rather than final confirmation sets. No runtime
threshold is fitted on either. A new untouched, speaker-disjoint confirmation
set is required before release.

The replacement British confirmation manifest is
`data/manifests/common-voice-british-confirmation-v0.1.jsonl` (SHA-256
`fe37d0c61e2fe28bd5f165415e69cc39c6be0d18bdaad67fed7b9d6466169308`). It
contains 100 CC0 Common Voice speakers and has zero speaker overlap with the
100-speaker set already used during development. Its labels remain uninspected
until a candidate has passed the regression gates.
