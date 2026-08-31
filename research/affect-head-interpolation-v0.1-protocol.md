# Affect head interpolation protocol

This is a bounded release-candidate experiment, not a sealed benchmark. RAVDESS has
already been opened during development, so its result may guide the engineering
decision but cannot support an unbiased generalization claim.

The experiment blends only the eight affect logits from the existing v0.9 and
v1.1 checkpoints. It does not retrain or alter the encoder, ASR, event head, style
head, VAD head, or OOD head.

- Evaluate alpha values from 0.00 through 0.50 in increments of 0.05, where
  `blended = (1 - alpha) * v0.9 + alpha * v1.1`.
- Fit APS-constrained calibration on the shared development rows for each alpha.
- Evaluate each fixed calibration on the shared RAVDESS rows.
- A candidate is eligible only when development macro-F1 is at least 0.64,
  RAVDESS macro-F1 is at least 0.40, APS is positive on both sets, coverage is at
  least 0.50 on both sets, and selective risk improves on both sets.
- Select the eligible candidate with the highest harmonic mean of development and
  RAVDESS macro-F1. Break ties in favour of the smaller alpha.
- If no candidate is eligible, stop affect work for v0.1 and mark categorical
  affect experimental. Do not start another broad training run.

Any selected candidate still requires the existing sealed-test and real-audio
release gates before it can be described as validated.

## Result

The aggregate grid selected `alpha = 0.50`. It reached macro-F1 0.6816 on the
shared development rows and 0.4451 on opened RAVDESS, with positive APS and
improving selective risk on both sets. The dual branch adds approximately 0.3
million parameters and preserves the shared encoder.

The aggregate pass did not survive a concrete transition check. On an internal
concatenation of one RAVDESS joy clip followed by one distress clip, the compact
ensemble assigned fear to both windows. The emotion2vec+ teacher assigned joy
and distress respectively to the two original clips. The ensemble is therefore
retained as a reproducible experiment but is not promoted into the v0.1
research preview. The v0.9 affect head remains the preview component, and the
next affect iteration should test teacher distillation rather than another
unfocused head continuation.
