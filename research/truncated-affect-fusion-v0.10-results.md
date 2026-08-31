# Truncated affect fusion v0.10

This sprint tested whether a compact acoustic encoder could replace the weak
affect behaviour in the Cadence v0.1 preview without changing its ASR or event
paths. The retained candidate combines the existing 241,904,650-parameter
Cadence graph with the reachable frontend and first three blocks of
emotion2vec+, followed by a 205,512-parameter affect head. The combined active
model has 299,842,014 parameters.

The affect head was trained on speaker-disjoint CREMA-D perceptual labels and
BERSt actor-prompt labels. Depths two and three, a linear head, two learning
rates, and seeds 42, 7, 17, and 73 were compared. Selection used the mean of
CREMA-D and BERSt development macro-F1 only. Seed 42 at depth three was retained;
RAVDESS was inspected only after that selection.

The fixed deployment fusion assigns 0.97 probability weight to the truncated
emotion2vec student and 0.03 to the existing Cadence affect head. The weight was
selected on the CREMA paired and BERSt development scores. It produced:

| Evaluation | Cadence v0.1 | v0.10 fusion |
|---|---:|---:|
| CREMA paired development macro-F1 | 0.5792 | 0.6713 |
| BERSt development macro-F1 | 0.1760 | 0.3009 |
| RAVDESS external macro-F1 | 0.3853 | 0.8133 |

Both predeclared RAVDESS controls pass: the high-intensity happy recording
scores joy above fear, and the high-intensity sad recording scores distress
above fear. An integrated eight-second transition made by concatenating those
two untouched clips produced a joy span from 0–4,000 ms and a distress span
from 4,000–8,075 ms. The whole-utterance classifier abstained because the clip
contains conflicting delivery, which is the intended behaviour.

Warm CPU inference for that 8.075-second composite took 1.99 and 1.76 seconds
in two consecutive runs (real-time factors 0.247 and 0.217). Model-loading time
is excluded because it is paid once when the service starts.

A follow-up batch-invariance audit reran all 480 RAVDESS clips one at a time.
The student and fusion macro-F1 scores remained 0.8154 and 0.8133, and both
exact controls still passed. Deployment therefore uses singleton affect
inference so padding from an unrelated clip cannot alter confidence values.

On 100 external British Common Voice clips, the student top category was
neutral for 82, other for 12, and a named non-neutral category for 6. These
clips do not have listener affect labels, so this is a distribution diagnostic,
not an accuracy result. Four neutral TTS voices speaking the hostile sentence
used in the lexical-leakage control produced only neutral or other as their top
category; none followed the angry wording. The browser omits neutral spans from
transcript highlighting.

The candidate is retained for the next release, but it is not published yet.
BERSt clears its fixed 0.30 gate narrowly, the current service has no validated
OOD estimator for the new branch, and broader microphone/accent checks and
quantization must precede a public replacement of v0.1. The reachable encoder
weights have been packaged independently of the 1 GB teacher checkpoint: the
package contains 57,731,852 backbone parameters in a 230,990,010-byte file and
reproduces the full teacher-backed predictor exactly.

The reproducible selection report is
`artifacts/evaluation/truncated-affect-fusion-v0.10/report.json`. The selected
head is `artifacts/training/truncated-emotion2vec-affect-v0.9/head.pt`.
The compact backbone is `artifacts/models/cadence-affect-student-v0.10/model.pt`.
