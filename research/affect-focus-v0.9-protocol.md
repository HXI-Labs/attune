# Affect focus v0.9 protocol

The upper-two v0.6 candidate improved affect after adapting shared acoustic
representations, but its multi-task sampler balanced seven corpora equally.
Only three corpora contained affect targets, so most sampled rows could not
update the affect objective. Development and opened-regression macro-F1 pass,
while the external RAVDESS result remains below the declared 0.40 floor.

Version 0.9 starts from the completed v0.8 checkpoint and samples only rows
with affect supervision. Corpus balancing gives CREMA-D, SUBESCO, and Thorsten
equal total sampling mass. The SenseVoice encoder, isolated ASR route, temporal
event head, style branch, pooling module, and all auxiliary heads remain
frozen. Only `affect_projection` and `affect_head` are trainable. The paired
same-text delivery loss remains active for eligible CREMA-D examples.

The run allows six epochs of 4,096 sampled rows, uses physical batches of six
with four-step accumulation, and stops after three stale validation epochs.
The head learning rate remains `2e-5`; no regression or external score is used
to choose a checkpoint or threshold.

The candidate is accepted only if development affect macro-F1 is at least
0.64, opened-regression macro-F1 stays within one absolute point of 0.6224,
RAVDESS improves without a dominant-class collapse, APS remains positive, and
all non-affect checkpoint tensors are byte-identical to v0.8. RAVDESS is an
opened diagnostic, not final confirmation. The fresh British set remains
sealed regardless of this experiment's result.

## Result

Epoch 6 was selected at validation loss 0.9416. The run trained 296,584
parameters. The selected checkpoint has SHA-256
`3e47e9b7974718f2589e0623d40e61e4c7eb6116d614e234694b52866f686e08`;
eight tensors changed, all under `affect_projection` or `affect_head`. Its ONNX
export has SHA-256 `5ffbada8fac7b3a5a298e21be03d45039de4072cc6d8c14ab6395acb01785d8e`
and maximum parity error `6.64e-05`. CTC logits are bit-identical to v0.8 on
three transcript-bearing clips.

| Evaluation | Macro-F1 | Brier | ECE | APS |
|---|---:|---:|---:|---:|
| Development | 0.6643 | 0.4503 | 0.0323 | +0.4500 |
| Opened regression | 0.6336 | 0.4861 | 0.0237 | +0.5182 |
| Opened RAVDESS | 0.3853 | 0.7695 | 0.2315 | +0.4089 |

The experiment passes its bounded improvement criteria and v0.9 becomes the
retained research candidate. RAVDESS improves by 4.7 macro-F1 points without a
single-class collapse, but remains 1.5 points below the 0.40 release floor.
This is evidence that affect-specific sampling was useful, not evidence that
the model is ready for publication. The fresh confirmation set remains sealed.
