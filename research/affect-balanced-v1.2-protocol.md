# Affect balanced v1.2 protocol

Affect paired v1.1 established that guaranteed same-text delivery pairs improve
the uncalibrated Acoustic Preference Score from -0.017 to +0.111 and retain an
external RAVDESS macro-F1 of 0.423. It did not pass the development gate.
Class-bias calibration changed class rankings and reduced APS to -0.131, while
temperature-only calibration preserved APS but reached only 0.613 macro-F1.
An APS-constrained bias scale selected on development data reached 0.624
macro-F1 with APS +0.030. All variants remain below the fixed 0.64 floor.

The remaining sampling imbalance is within corpora. CREMA-D listener targets
have six supported hard-label modes, but neutral is the argmax for most clips.
The previous sampler gave each corpus equal total mass while retaining this
within-corpus imbalance. Version 1.2 gives each observed hard-label mode equal
mass within its corpus and keeps equal total mass across corpora. The audited
weight totals are 1/6 for each CREMA-D mode, 1/7 for each SUBESCO mode, and 1/5
for each Thorsten mode. Soft listener distributions remain the supervised
targets; argmax is used only to construct sampling weights.

Training restarts from retained v0.9, not rejected v1.1, and reuses the
15,389-row manifest with SHA-256
`329be1a98e632b91823d15eacb8762990dcd29050acd28aa777d9d590f1905fb`.
All batches must contain a dataset-scoped same-text, different-delivery pair.
Only `affect_projection` and `affect_head` may change. The paired loss remains
0.2, the head learning rate remains `2e-5`, and the run permits six epochs of
8,192 samples with four-step gradient accumulation.

Development calibration uses the fitted temperature and regularized class
bias, then searches bias scales from 0 to 1 in increments of 0.05. Scales with
non-positive development APS are excluded; among the remaining scales, the
one with highest development macro-F1 is selected. This procedure is fixed
before external evaluation and recorded in the calibration bundle. If no
scale preserves positive APS, the candidate fails without opening additional
data.

The candidate is retained only if all of the following hold:

- Development macro-F1 is at least 0.64 with positive APS.
- Opened-regression macro-F1 is at least 0.6236 with positive APS.
- Opened RAVDESS macro-F1 is at least 0.40 with positive APS and no predicted
  class above 45%.
- Selective risk improves on every evaluated affect split.
- Every epoch reports an active paired-batch fraction of 1.0.
- Only the eight permitted affect tensors differ from v0.9.
- CTC logits remain exactly equal to v0.9.

Development alone selects the epoch and calibration. Opened regression and
RAVDESS only accept or reject the completed candidate. The fresh British
confirmation set remains sealed.

## Result

Epoch 5 was selected at validation loss 1.1126. All six epochs achieved an
active paired-batch fraction of 1.0. The selected checkpoint has SHA-256
`83bcb2e90a5e258675f6bc8ef4e023661a7c2641394533c73c770be39cecd0cc`;
its ONNX export has SHA-256
`6d00b49f205bf4cd5f7e6bf689664ce6f8123a71051549a6d93e7c00fd404e0a`
and maximum parity error `6.64e-05`. The scope audit found exactly eight
changed affect tensors and no violations.

APS-constrained development calibration selected bias scale 0.25. The complete
2,157-row development evaluation reached macro-F1 0.6282, APS +0.0572, Brier
score 0.3009, and ECE 0.0959. Temperature-only calibration reached macro-F1
0.6277 with APS +0.1178. Selective risk improved and all 1,241 speech controls
had zero auxiliary false positives.

The candidate is rejected because neither calibration reaches the fixed 0.64
development macro-F1 floor. Opened regression, RAVDESS, and fresh British data
were not evaluated. Retained v0.9 remains unchanged. The result supports class
balancing but shows that sampling alone does not fully convert same-text
delivery contrasts into the required class decision changes.
