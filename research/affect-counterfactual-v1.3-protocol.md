# Affect counterfactual v1.3 protocol

This experiment is a predeclared fallback and runs only if affect-balanced v1.2
fails an acceptance gate. Version 1.1 showed that guaranteed same-text,
different-delivery batches restore positive uncalibrated Acoustic Preference
Score, but its paired objective only pushes affect embeddings apart. Separation
does not specify which affect logits should change or in which direction.

Version 1.3 replaces that embedding-repulsion term with a counterfactual
logit-ranking loss. For each unordered same-dataset, same-text pair `(i, j)`,
let `q` be the soft listener distribution and `z` the predicted affect logits.
For every class where `q_i - q_j` is non-zero, the loss is:

```text
abs(q_i - q_j) * softplus(
  0.25 - sign(q_i - q_j) * (z_i - z_j)
)
```

The loss is normalized by the total absolute target difference. It directly
requires the prediction to move in the listener-supported direction while
cancelling any logit contribution shared by identical lexical content. The
ordinary soft-label cross-entropy remains the primary objective. Pair IDs are
dataset-scoped, each physical batch contains a valid contrast, and unordered
pairs are counted once.

The run starts from retained v0.9 and uses the class-balanced sampler and
15,389-row manifest defined for v1.2. The manifest SHA-256 remains
`329be1a98e632b91823d15eacb8762990dcd29050acd28aa777d9d590f1905fb`.
The counterfactual loss weight is `0.2`; the previous embedding-repulsion weight
is zero. All other training settings remain fixed: six epochs, 8,192 samples
per epoch, batches of six, four-step gradient accumulation, a `2e-5` head
learning rate, and seed 42. Only `affect_projection` and `affect_head` may
change.

Development selects the checkpoint and APS-constrained calibration. Opened
regression and RAVDESS sets only accept or reject a candidate that first passes
development. The fresh British confirmation set remains sealed. The candidate
is retained only if all of the following hold:

- Development macro-F1 is at least 0.64 with positive APS.
- Opened-regression macro-F1 is at least 0.6236 with positive APS.
- Opened RAVDESS macro-F1 is at least 0.40 with positive APS and no predicted
  class above 45%.
- Selective risk improves on every evaluated affect split.
- Every epoch reports an active paired-batch fraction of 1.0.
- Only the eight permitted affect tensors differ from v0.9.
- CTC logits remain exactly equal to v0.9.

No margin, loss weight, epoch, checkpoint, or calibration parameter will be
selected using the opened external sets.

## Result

Epoch 5 was selected at validation loss 1.1361. All six epochs achieved an
active paired-batch fraction of 1.0. The selected checkpoint has SHA-256
`a62699df05597b85d3d08320720794337b15dae23f4624de2102d517ece3b6c0`;
its ONNX export has SHA-256
`c6f505f9fea25678a2a384b7b01db98d8fd854d7e172cc3395b435711aa359ab`
and maximum parity error `6.64e-05`. The scope audit found exactly eight
changed affect tensors and no violations.

APS-constrained development calibration selected bias scale 0.25. The complete
2,157-row development evaluation reached macro-F1 0.6267, APS +0.0303, Brier
score 0.3005, and ECE 0.0951. Temperature-only calibration reached macro-F1
0.6249 with APS +0.1077. Selective risk improved and all 1,241 speech controls
had zero auxiliary false positives.

The candidate is rejected because it misses the 0.64 development macro-F1
floor and underperforms v1.2. Opened regression, RAVDESS, and fresh British data
were not evaluated. Retained v0.9 remains unchanged. Direct logit ranking did
not improve the quality-acoustic-reliance trade-off at the predeclared weight;
further affect work should add better natural perceptual coverage rather than
tune this objective against the same development set.
