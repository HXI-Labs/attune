# Affect consensus v0.11

This sprint improves the compact acoustic affect branch without changing
Cadence ASR, event detection, encoder depth, parameter count, or inference
latency. Two independently trained depth-three heads produce soft training
targets for one deployment head. The selected student uses 0.8 teacher
consensus and 0.2 listener-label supervision.

The teacher pair and distillation weight were selected on CREMA-D and BERSt
development data only. The fixed rule chooses the highest mean development
macro-F1 among candidates with BERSt macro-F1 of at least 0.30. RAVDESS was
evaluated after selection.

| Evaluation | v0.10 student | v0.11 consensus student |
|---|---:|---:|
| CREMA-D perceptual development macro-F1, 819 clips | 0.5399 | 0.5901 |
| BERSt development macro-F1, 487 clips | 0.2917 | 0.3042 |
| Mean development macro-F1 | 0.4158 | 0.4471 |
| RAVDESS external macro-F1, 480 clips | 0.8154 | 0.7981 |

The full CREMA-D improvement is 0.0502 macro-F1. A 10,000-resample paired
bootstrap gives a 95 percent interval of +0.0206 to +0.0802. The BERSt change
is +0.0124 with an interval of -0.0222 to +0.0477.

The fixed Cadence fusion was reselected on the shared paired-CREMA and BERSt
development clips. It assigns 0.86 probability weight to the v0.11 acoustic
student and produces:

| Evaluation | v0.10 fusion | v0.11 fusion |
|---|---:|---:|
| CREMA paired development macro-F1, 120 clips | 0.6713 | 0.6427 |
| BERSt development macro-F1 | 0.3009 | 0.2990 |
| RAVDESS external macro-F1 | 0.8133 | 0.8220 |

The narrow paired differences are not statistically resolved. The paired
95 percent intervals for v0.11 minus v0.10 are -0.0899 to +0.0301 on CREMA,
-0.0375 to +0.0346 on BERSt, and -0.0200 to +0.0366 on RAVDESS. The strict
legacy fusion gate remains false because BERSt is 0.0010 below its provisional
0.30 floor. That report has not been relaxed or rewritten.

Both predeclared transition controls pass. On the untouched 8.075-second
joy-to-distress composite, integrated inference emits joy from 0 to 4,000 ms
at 0.405 confidence and distress from 4,000 to 8,075 ms at 0.686 confidence.
The utterance-level output now abstains with `mixed_affect_spans`; it no longer
forces one category over a turn containing conflicting confident spans.

The four neutral system voices reading hostile text remain a lexical control.
The v0.11 student predicts neutral for three and other for one; anger
probability ranges from 0.0499 to 0.1282. No voice has anger as its top
category.

The active model remains 299,842,014 parameters. The compact acoustic backbone
contains 57,731,852 parameters and is byte-identical to v0.10 because only the
205,512-parameter head changed. The v0.11 backbone package is 230,990,010 bytes
with SHA-256
`0194733c5f35c2fbb1541528afe9cd1cc67b00a541f412e52b43cae0d82e99da`.
The student checkpoint SHA-256 is
`e201222bfe163ce438187b514dfa16e337cdbf28997dbdf5b1ff19f8e1964c38`.

Paired-intensity consistency, class-bias calibration, a fourth encoder block,
and local last-block fine-tuning were evaluated and rejected. They either
reduced development accuracy, weakened external generalisation, exceeded the
size limit, or were inefficient on the available CPU.

The reproducible training command is
`scripts/distill_truncated_emotion2vec_head.py`. The selection report is
`artifacts/evaluation/truncated-emotion2vec-affect-consensus-v0.11/report.json`,
the fusion report is
`artifacts/evaluation/truncated-affect-fusion-consensus-v0.11/report.json`, and
the packaged backbone is
`artifacts/models/cadence-affect-student-v0.11/model.pt`.

v0.11 is retained as the next development candidate, not published as a public
replacement. The remaining blockers are a validated OOD estimator, broader
consented natural-speech and microphone evaluation, and quantization of the
new acoustic branch.
