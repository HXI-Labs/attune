# Affect control fine-tuning v0.13

This experiment reduces unsupported named-affect output from the compact v0.11
candidate without changing Cadence ASR, event detection, style policy, encoder
depth, fusion weight, or parameter count. It starts from the v0.11 head and
applies 25 small updates. Each update preserves the incumbent distribution on
labeled replay examples and applies a low-weight loss that moves ordinary
Common Voice speech away from named affect categories. The control loss does
not force a hard neutral label; it only rewards combined `neutral` and `other`
probability mass.

The search used 479 Common Voice training controls and 62 disjoint development
controls. CREMA-D and BERSt remained the labeled development gates. The
selected candidate uses control weight 0.02, learning rate 0.0001, 25 steps,
and seed 42. Selection did not use RAVDESS or the separate 100-speaker British
inspection set.

A fixed-weight replication over seeds 7, 17, 42, and 73 searched only 5, 10,
15, 20, and 25 update steps. Every seed produced a candidate that passed the
same development constraints, reduced Common Voice development named tops to
1, retained both transition controls, and reached fused external RAVDESS
macro-F1 between 0.8322 and 0.8426. Seed 42 remains the declared release seed;
external performance was not used to replace it.

| Fixed Cadence fusion | v0.11 | v0.13 |
|---|---:|---:|
| Paired CREMA-D development macro-F1, 120 clips | 0.6427 | 0.6401 |
| BERSt development macro-F1, 487 clips | 0.2990 | 0.3070 |
| Mean development macro-F1 | 0.4708 | 0.4735 |
| RAVDESS external macro-F1, 480 clips | 0.8220 | 0.8322 |
| Common Voice development named top labels, 62 clips | 5 | 1 |
| British external named top labels, 100 clips | 7 | 3 |
| British external named labels emitted at 0.40 | not selected | 1 |

The fixed confidence threshold is now 0.40. On v0.13, increasing the threshold
from 0.25 to 0.40 changes paired CREMA-D coverage from 0.9667 to 0.7167 and
retained accuracy from 0.6466 to 0.7093. BERSt coverage changes from 0.7556 to
0.2012 and retained accuracy from 0.3342 to 0.3776. The lower BERSt coverage is
intentional: uncertain cross-domain predictions should abstain.

Both predeclared RAVDESS transition controls pass. Full integrated inference
on the untouched 8.075-second composite emits joy from 0 to 4,000 ms at 0.407
confidence and distress from 4,000 to 8,075 ms at 0.651 confidence. The global
result abstains with `mixed_affect_spans`.

The four neutral system voices reading the hostile regression sentence are
all transcribed correctly. They emit no events or styles. Two return neutral
at 0.420 and 0.434; two abstain at 0.323 and 0.319. This remains a lexical
control rather than human affect ground truth.

The active composition remains 299,842,014 parameters. The packaged acoustic
backbone contains 57,731,852 parameters, is 230,990,010 bytes, and has SHA-256
`0194733c5f35c2fbb1541528afe9cd1cc67b00a541f412e52b43cae0d82e99da`.
The reproducibly selected head checkpoint has SHA-256
`baf32e3a5d9ae2e74a0e263cb0c5fae9ee197d5da8c3edcb9d6918e640f208e3`.

Dynamic INT8 was first evaluated with PyTorch QNNPACK over the compact branch's
linear layers. Fused RAVDESS macro-F1 is 0.8304, an absolute loss of 0.0019
from FP32 and within the two-point gate. Branch inference real-time factor is
0.0586 on the development Mac. On the external British controls, named top
labels change from 3 to 2 and one label clears 0.40 in both modes.

The retained portable artifact is a standalone QNNPACK TorchScript graph. It
contains the truncated frontend, three transformer blocks, pooling, and affect
head without requiring the FunASR model loader. It is 81,796,647 bytes and has
SHA-256
`9be218fa538caad36b397fe25230d55b2b2904a5624850a6ab1b09a508e9a506`.
Fresh-process validation is bit-identical to runtime dynamic INT8 on mixed
lengths. Before post-quantization calibration it reaches 0.8114 student and
0.8311 fused macro-F1 on all 480 external RAVDESS clips, an absolute fused loss
of 0.0011 from FP32, at real-time factor 0.0727.

The post-quantization transform was fitted on soft listener targets with
corpus-balanced loss. Five-fold evaluation groups clips by speaker. Across 607
development clips it raises portable INT8 macro-F1 from 0.3544 to 0.3702,
compared with 0.3766 for FP32. Cross-validated Brier score is 0.6724 versus
0.6722 for FP32, and negative log-likelihood is 1.6656 versus 1.6715. The final
2,818-byte calibration has SHA-256
`a9ff1e51feeb706170c6aa8ab75b2fa05d6734dd731b4428c4354880cb72500d`
and records the model hash; runtime loading fails if the two artifacts do not
match.

The calibrated artifact reaches 0.8694 fused macro-F1 on the separate RAVDESS
set at real-time factor 0.0720. Both predeclared transition controls pass.
Integrated inference emits joy from 0 to 4,000 ms at 0.490 and distress from
4,000 to 8,075 ms at 0.557, then abstains at utterance level with
`mixed_affect_spans`.

The four regenerated neutral voices reading the exact hostile sentence are
transcribed correctly through the final portable composition. All four emit no
events or styles. Daniel, Moira, and Samantha return neutral at 0.581, 0.448,
and 0.622; Tessa abstains below the 0.40 threshold. None invents anger,
whispering, coughing, or sneezing. These clips remain lexical controls rather
than human affect ground truth.

The first direct ONNX trace was rejected because it dropped the padding mask
and drifted from PyTorch. A corrected trace retains explicit `samples` and
`padding_mask` inputs and matches FP32 within 0.000001. Selective MatMul/GEMM
quantization produces an 82,060,030-byte graph with SHA-256
`682123a730186981e336bcd797b2311bee22cb234148638f1b0591a56f638bd8`.
It reaches 0.8314 fused RAVDESS macro-F1, but stricter evaluation over the 607
development clips yields 0.3493 versus 0.3766 for FP32, missing the two-point
parity gate. It is retained as a research export rather than the deployment
default. Quantizing the convolutional frontend was also tested and rejected
because it caused substantial probability drift.

The reproducible training entry point is
`scripts/fine_tune_truncated_affect_controls.py`. The selected checkpoint is
`artifacts/training/truncated-emotion2vec-affect-control-finetune-v0.13/head.pt`,
the report is
`artifacts/evaluation/truncated-emotion2vec-affect-control-finetune-v0.13/report.json`,
and the compact package is `artifacts/models/cadence-affect-student-v0.13/`.
Portable export and evaluation reports are stored in the v0.13 evaluation
directory. The retained reports are
`portable-torchscript-int8-calibration-fitted.json` and
`portable-torchscript-int8-calibrated-ravdess.json`. Export and evaluation use
`scripts/export_truncated_affect_torchscript.py`,
`scripts/evaluate_portable_affect_calibration.py`, and
`scripts/evaluate_portable_affect.py`.

v0.13 is the retained accuracy candidate. It is not yet a public replacement
for v0.1. A fresh consented human hostile-speech recording and broader natural
listener-labeled evaluation remain release blockers. The post-export
calibration gate is complete.
