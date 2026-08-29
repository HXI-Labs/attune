# Project Attune v0.1 technical report

## Abstract

Project Attune v0.1 tests whether a compact pretrained speech model can retain
ordinary transcription while adding time-aligned vocal events, delivery styles,
perceived affect, uncertainty, and abstention. The final local candidate is a
241.6M-parameter SenseVoice-Small derivative. It shares most of the encoder,
uses an adapted perception path, and preserves CTC ASR through frozen copies of
two upper encoder blocks. A mixed-precision ONNX graph quantizes most eligible
shared weights to dynamic per-channel INT8 while retaining sensitive tails and
heads in FP.

On a 401-row source-labelled evaluation partition, FP/mixed-INT8 WER is
0.0734/0.0782. Under a precision-first confidence floor, localized-event
segment macro-F1 is 0.6645/0.6548 and auxiliary false-positive rate across 189
ordinary-speech controls is 0.0/0.0. Offline
event-presence macro-F1 is 0.8258/0.8220, style macro-F1 is 1.0/1.0,
supported-class affect macro-F1 is 0.4746/0.4591, and OOD F1 is
0.9907/0.9747. Acoustic Preference Score is +0.2727. Development-selected
abstention reduces FP affect error from 0.4773 at full coverage to 0.4128 at
0.8258 coverage. The mixed graph runs at 0.0534 real-time factor on the
development Mac CPU with p95 latency of 415 ms and 100% structurally valid JSON
and XML.

These results establish a reproducible technical prototype, not a release or
broad emotion understanding. Affect data is acted and one-hot, only three
events have strong temporal labels, weak event/style outputs are disabled,
V/A/D is unavailable, and no downstream human study has been run. An initial
sealed pass triggered an ASR-only architectural correction, and a later live
false-positive failure triggered conservative runtime gating. The corrected
ASR and auxiliary policy need confirmation on new untouched audio.

## 1. Problem formulation

Transcript-only voice pipelines discard audible evidence such as laughter,
coughing, whispering, shouting, and crying-related delivery. Attune represents
observable vocal behaviour separately from a probabilistic affect
interpretation. It does not claim access to internal emotion.

The system output contains:

1. transcript text and CTC token spans grouped by SentencePiece word boundaries;
2. bounded vocal events where strong temporal evidence exists;
3. research-only weak event-presence logits, disabled in runtime;
4. research-only delivery-style logits, disabled in runtime;
5. a categorical perceived-affect distribution;
6. calibrated abstention and OOD information; and
7. audio-quality and model metadata in schema-v2 JSON.

XML is a deterministic view. Transcript content and model metadata remain
separate trusted fields to prevent spoken markup from becoming an instruction.

## 2. Claim boundary

The v0.1 question is narrow: can a compact, locally deployable model recover
useful paralinguistic evidence without sacrificing ASR or structural
reliability? It does not test clinical inference, deception, intent, protected
traits, or high-stakes decisions. It also does not test the original downstream
interaction hypothesis because a controlled human response study is still
future work.

## 3. Data

The audited bundle contains 2,490 rows and 3.79 hours:

| Source | Rows | Role | Limitation |
|---|---:|---|---|
| Common Voice 17 English | 600 | CTC replay | transcript only |
| DCASE 2016 Task 2 | 316 | strong event timing/presence | synthetic scenes |
| FSD50K bounded slice | 320 | weak events/styles | clip labels, limited speaker metadata |
| VocalSound | 762 | weak vocal-event presence | acted isolated sounds |
| CREMA-D paired v0.1 | 492 | categorical affect and lexical conflict | acted one-hot labels |

Splits contain 1,691 train, 398 development, and 401 evaluation rows. Known
speakers and DCASE recording groups are disjoint. FSD50K clip identity is
disjoint, but speaker disjointness cannot be established from missing metadata.
Missing task labels are masked rather than converted to negatives.

Only `laugh`, `cough`, and `throat_clear` have strong frame targets. Other
events have presence supervision only. `shouting` and `whispering` are
utterance-scope weak styles. CREMA-D provides the six supported affect targets:
`neutral`, `joy`, `distress`, `anger`, `fear`, and `other`. It does not provide
multi-rater distributions. `surprise`, `ambiguous`, and V/A/D do not have
reviewed positive supervision.

The feature manifest SHA-256 is
`00fded47e4bb177af4816cc07cd8939f8376e6c77c423e17cd7624789c03c708`.

## 4. Model

### 4.1 Shared perception model

SenseVoice-Small supplies the acoustic frontend, encoder, and CTC output layer.
Attune adds:

- a convolutional frame head for multi-label event activity and boundaries;
- an utterance event-presence head;
- attentive mean/standard-deviation pooling;
- an affect-specific projection;
- independent sigmoid style outputs;
- a categorical affect softmax;
- an OOD logit and normalized OOD embedding; and
- a dormant three-dimensional output head.

The frozen probe updates 1.29M task-head parameters. The selected adaptation
updates the task heads, upper two encoder blocks, and final norms: 7.61M
trainable parameters.

### 4.2 ASR split tail

The initial adapted model shared one final encoder representation between ASR
and perception. Development WER degradation was within the fixed one-point
gate, but the first evaluation pass showed 2.23 absolute points of degradation.

The release architecture shares the lower 47 encoder blocks, then computes two
views:

- the adapted final two blocks for perception; and
- frozen copies of the original final two blocks for CTC ASR.

Both pass through the same frozen temporal-predictor block parameters; only
their computation is separate. Frozen copies of final norms preserve the base
ASR representation. This adds 6.32M parameters, resulting in 241,609,098 total.
Development confirms exact frozen-base WER and unchanged perception metrics.

## 5. Training

The multi-task loss combines CTC, class-balanced frame event activity, boundary
losses, weak event presence, style BCE, categorical affect cross-entropy,
paired acoustic contrast, and augmentation consistency. Task masks ensure that
absent annotations do not create false negatives. Corpus-aware duration batches
prevent large sources from suppressing rarer tasks.

The frozen-head candidate selected epoch 7 but failed the affect development
floor at 0.2782 macro-F1. The upper-two run warm-started only those selected
task heads. It ran for 14 epochs, early-stopped after three stale epochs, and
selected epoch 11 with development loss 1.2537. Its delta SHA-256 is
`54f742a9136e43c9dee29532a719ee170bd8f65e8ab68f8bad542d7a9364ec46`.
Both runs completed locally on CPU; no cloud GPU spend was required.

## 6. Calibration and evaluation

Temperatures and class-specific thresholds are fit only on development scores.
FP and mixed INT8 receive separate calibration bundles. Evaluation then uses
fixed thresholds. Runtime applies a 0.98 localized-event confidence floor and
empty allowlists for weak event-presence and style heads.

Metrics include WER, frame and segment event F1, temporal IoU and boundary
error, weak-presence/style F1, categorical affect F1/Brier/ECE, selective risk,
OOD F1, and Acoustic Preference Score. The runtime harness validates the input
contract before balanced sampling and separately reports incompatible source
rows.

Release floors were fixed before the final run:

- no more than 0.01 FP WER degradation from the base route;
- FP localized event ≥0.50, presence ≥0.50, style ≥0.60, affect ≥0.40, OOD
  ≥0.75, coverage ≥0.50, and APS >0;
- mixed INT8 perception/OOD losses ≤0.02 and WER loss ≤0.005;
- CPU RTF ≤1.0, JSON/XML validity 1.0, and committed retraction 0.
- ordinary-speech auxiliary false-positive rate ≤0.01, localized false events
  ≤0.10 per minute, and a mandatory hostile-speech real-audio regression.

## 7. Results

### 7.1 Development

The split-tail FP graph obtains WER 0.1110, event segment F1 0.8918,
event-presence F1 0.8715, style F1 1.0, affect F1 0.4685, OOD F1 0.9931,
coverage 0.80, and APS +0.38. The mixed graph obtains WER 0.1127, event segment
F1 0.9138, event presence 0.8818, style 1.0, affect 0.4902, OOD 0.9954,
coverage 0.80, and APS +0.42.

### 7.2 Evaluation partition

| Metric | FP | mixed INT8 | Delta |
|---|---:|---:|---:|
| WER | 0.0734 | 0.0782 | +0.0048 |
| Event frame macro-F1 at >= 0.98 | 0.4921 | 0.4887 | -0.0034 |
| Event segment macro-F1 at >= 0.98 | 0.6645 | 0.6548 | -0.0097 |
| Speech-control auxiliary false-positive rate | 0.0000 | 0.0000 | 0.0000 |
| Speech-control localized false events/minute | 0.0000 | 0.0000 | 0.0000 |
| Event presence macro-F1 | 0.8258 | 0.8220 | -0.0038 |
| Style macro-F1 | 1.0000 | 1.0000 | 0.0000 |
| Affect macro-F1 | 0.4746 | 0.4591 | -0.0155 |
| Affect ontology macro-F1 | 0.3560 | 0.3443 | -0.0116 |
| OOD F1 | 0.9907 | 0.9747 | -0.0159 |
| Affect coverage | 0.8258 | 0.8712 | +0.0455 |
| APS | +0.2727 | +0.2727 | 0.0000 |

FP affect Brier score is 0.5999 and ECE is 0.1134. Selective error is 0.4128
versus 0.4773 at full coverage. Automatic gates pass, while the manual
hostile-speech retest deliberately remains false.

## 8. Error analysis

The detailed report is `research/error-analysis/v0.1-fp-sealed.md`.

Among 189 ASR-scored clips, 70 contain at least one word edit. The largest
errors include highly expressive CREMA-D takes and proper-name/normalization
errors in Common Voice. Localized event performance is:

| Label | Segment F1 | TP | FP | FN | Boundary MAE |
|---|---:|---:|---:|---:|---:|
| laugh | 0.8889 | 36 | 0 | 9 | 235.0 ms |
| cough | 0.3860 | 11 | 2 | 33 | 182.7 ms |
| throat_clear | 0.7188 | 23 | 0 | 18 | 157.8 ms |

Cough is the weakest localized label because it produces both more insertions
and misses. The highest-confidence categorical errors cross joy/anger,
distress/other, and anger/neutral. Since targets are acted one-hot source
labels, these confusions are evidence about corpus classification—not proof of
incorrect access to an internal state. FP OOD produces four errors.

## 9. Deployment

FP export checks ten floating outputs with maximum absolute PyTorch/ONNX error
of `8.39e-05`. The graph is 968.7 MB.

Dynamic per-channel QInt8 is applied to most eligible shared linear weights.
The upper eight perception blocks, frozen ASR tail, pooling, and utterance heads
remain FP. The resulting graph is 594.8 MB (size ratio 0.614). This is a
mixed-precision INT8 deployment artifact, not a fully integer graph.

On the development Mac CPU, 40 valid clips / 169.0 seconds of audio run at
0.0534 RTF. p50/p95/p99 per-clip latency is 206/415/440 ms. JSON and XML
validity are both 100%. The service smoke validates batch HTTP inference, three
provisional WebSocket revisions, one committed result, and zero errors.

## 10. Validity threats

The main threats are narrow and artificial data, acted affect, weak labels,
small semantic-conflict groups, unsupported classes, and missing naturalistic
V/A/D. Style performance is especially likely to overstate real-world
generality. Accent, device, noise, culture, and vocal-presentation fairness are
not sufficiently established.

Archived unmodified-base measurements report WER 0.2365 on 100 Ghanaian-English
broadcast clips and 0.1149 on 100 British Common Voice speakers. The final FP
route uses exact frozen copies of that base tail, but the final graph and its
mixed-INT8 variant were not rerun because the audio caches were not retained.
Those numbers are therefore contextual baselines, not final-model accent
results.

The initial evaluation partition was correctly left untouched for perception
selection, but it exposed the ASR failure that motivated the split-tail route.
Although that route is mechanically guaranteed to restore the base FP ASR path
and was validated on development, its corrected score was measured on the same
already-opened partition. A new external untouched test is therefore required
for a confirmatory ASR claim.

## 11. Ethics and release

Attune output must be phrased as observable or uncertain perception. It must
not be used for diagnosis, lie detection, protected-trait inference, covert
surveillance, or automated high-stakes decisions. Audio, third-party weights,
and derivative weights are not made public until their independent licence,
consent, and misuse reviews permit release.

No downstream interaction benefit is claimed. A preregistered human evaluation
should compare transcript-only, hard-label, and calibrated structured contexts
while measuring usefulness, restraint, overreaction, and patronizing behaviour.

## 12. Conclusion

Attune v0.1 demonstrates that a sub-300M speech model can preserve base ASR and
add useful localized and utterance-level paralinguistic outputs with calibrated
abstention, positive acoustic preference, structurally safe serialization, and
near-real-time CPU deployment. The result is a credible research-engineering
foundation. Its strongest contribution is the complete, failure-aware
technical pipeline—not a claim that compact models now understand human
emotion in the wild.
