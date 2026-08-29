# Project Attune Technical Completion Report

- **Project:** Beyond the Transcript — Compact, Time-Aligned Paralinguistic
  Transcription for Emotion-Aware Voice Interaction
- **Working codename:** Project Attune
- **Owner:** Jerry Buaba / HXI Labs
- **Report date:** 29 August 2026
- **Release:** Technical v0.1
- **Status:** Technical model and deployment release completed; future human
  and naturalistic-data studies remain outstanding

## 1. Executive summary

Project Attune now has a working compact speech model that combines ordinary
speech transcription with time-aligned vocal-event detection, delivery-style
classification, perceived-affect estimation, uncertainty, out-of-distribution
detection, and calibrated abstention.

The final model is a 241,609,098-parameter derivative of SenseVoice-Small. It is
available as both a full-precision ONNX graph and a smaller mixed-precision INT8
deployment graph. The final candidate passes all 20 executable technical
release gates covering model size, ASR preservation, event and style quality,
affect performance, semantic-acoustic preference, uncertainty, quantization,
runtime, output validity, and streaming stability.

The deployed system accepts 16 kHz mono PCM16 WAV audio and returns authoritative
schema-v2 JSON. It can also generate deterministic, injection-safe XML. Both a
batch HTTP API and pseudo-streaming WebSocket API have been implemented and
tested against the real final model.

The work completed here should be considered the technical v0.1 research
release. It does not yet establish that Attune improves conversational responses
in a controlled human study. It also does not provide reviewed valence,
arousal, and dominance outputs because a suitable licensed dimensional-affect
training source was not available.

## 2. Final system delivered

The completed system produces:

- English speech transcription using the SenseVoice CTC path.
- CTC-derived word timestamps rather than fabricated uniform timing.
- Frame-level and segment-level localization for strongly supervised vocal
  events.
- Utterance-level event-presence output for weakly supervised labels.
- Delivery-style predictions for shouting and whispering.
- A probability distribution over perceived affect categories.
- Confidence-aware affect abstention.
- Out-of-distribution detection.
- Model, audio, language, quality, timing, and uncertainty metadata.
- Valid schema-v2 JSON.
- Deterministic XML rendered only from validated JSON.
- Offline batch inference.
- A local batch API.
- Pseudo-streaming inference with provisional and immutable committed states.
- A mixed-precision INT8 deployment model.

The system deliberately does not claim to identify a speaker's true emotional
state. Affect output is represented as uncertain perceived expression.

## 3. Data preparation and governance

A checksum-verified, multi-corpus source-labelled bundle was prepared for joint
training and evaluation.

### 3.1 Final bundle

| Property | Value |
|---|---:|
| Total rows | 2,490 |
| Total duration | 3.79 hours |
| Training rows | 1,691 |
| Development rows | 398 |
| Evaluation rows | 401 |
| Known speaker overlap | 0 |
| Feature-manifest SHA-256 | `00fded47e4bb177af4816cc07cd8939f8376e6c77c423e17cd7624789c03c708` |

### 3.2 Sources and roles

| Source | Main use | Important limitation |
|---|---|---|
| Common Voice 17 English | CTC ASR replay | Transcript supervision only |
| DCASE 2016 Task 2 | Strong laugh, cough, and throat-clear timing | Synthetic acoustic scenes |
| FSD50K bounded subset | Weak event and style supervision | Clip-level labels and incomplete speaker metadata |
| VocalSound | Vocal-event presence | Acted isolated sounds; no speech-localization target |
| CREMA-D paired subset | Affect and semantic-acoustic conflict pairs | Acted, one-hot categorical labels |

Known speakers and source-recording groups were separated across splits where
the source metadata permitted it. Missing task annotations were masked during
training rather than silently treated as negative examples.

Every source has a provenance record covering its revision, licence status,
redistribution restrictions, and intended use. Audio remains local and is not
committed to the repository.

## 4. Schema and trusted output channel

A new schema-v2 output contract was implemented. It separates:

- Transcript text and word timing.
- Localized vocal events.
- Utterance-scope event evidence.
- Delivery styles.
- Affect probabilities.
- Dimensional availability.
- Abstention and OOD state.
- Audio and language metadata.
- Provisional and committed streaming state.

The JSON schema is the source of truth. XML is generated deterministically and
escapes transcript content, preventing spoken text from injecting markup.
Transcript text and model-produced metadata also remain separate fields when
passed to downstream systems.

Structural tests cover invalid timestamps, probability consistency, overlapping
spans, XML escaping, schema validity, and migration from the earlier schema.

## 5. Model architecture

### 5.1 Perception path

SenseVoice-Small supplies the acoustic frontend, shared encoder, and CTC
vocabulary. Attune adds:

- A convolutional temporal event head.
- Event-start and event-end outputs.
- An utterance event-presence head.
- Attentive statistical pooling.
- An affect-specific projection.
- Independent multi-label style outputs.
- A categorical affect head.
- An OOD logit and normalized OOD embedding.
- A dormant V/A/D head retained for future reviewed dimensional data.

### 5.2 ASR-preserving split tail

The initially selected upper-two model passed the perception gates, but the
first evaluation pass exposed excessive ASR degradation: its WER was 2.23
absolute points worse than the frozen SenseVoice route. The release gate was
not lowered.

Instead, a compact split-tail architecture was implemented:

- The lower 47 encoder blocks remain shared.
- The adapted final two blocks drive the paralinguistic perception heads.
- Frozen copies of the original final two blocks drive CTC ASR.
- The frozen downstream temporal-predictor parameters remain shared.

This added 6,318,080 parameters and restored full-precision ASR to the frozen
base result while leaving the selected perception outputs unchanged. The final
model contains 241,609,098 parameters, remaining below the 300M project limit.

## 6. Training programme completed

### 6.1 Frozen-head probe

The first candidate trained only the new task heads.

- Selected epoch: 7.
- Development loss: 4.4326.
- Localized-event segment F1: 0.9074.
- Event-presence F1: 0.8697.
- Style F1: 1.0000.
- OOD F1: 0.9977.
- Affect macro-F1: 0.2782.

The candidate was rejected because affect performance failed the fixed 0.40
development floor.

### 6.2 Upper-two adaptation

The second candidate warm-started only the previously trained task heads and
adapted the upper two encoder blocks.

- Trainable parameters: 7,609,931.
- Completed epochs: 14.
- Selected epoch: 11.
- Selected development loss: 1.2537.
- Early stopping: three stale epochs.
- Selected delta SHA-256:
  `54f742a9136e43c9dee29532a719ee170bd8f65e8ab68f8bad542d7a9364ec46`.

This candidate materially improved affect performance while retaining strong
event, style, OOD, and semantic-conflict results.

Training completed locally on CPU. No Vast.ai or RunPod instance was ultimately
required, and cloud training cost was £0.

## 7. Calibration and uncertainty

Separate calibration bundles were fitted for full precision and mixed INT8
using development scores only. The following were calibrated:

- Event-frame thresholds.
- Event-presence thresholds.
- Style thresholds.
- Affect temperature and abstention threshold.
- OOD temperature and threshold.
- Minimum event duration and short-gap bridging.

The final FP affect system covers 82.58% of supported affect clips. Error falls
from 47.73% at full coverage to 41.28% among retained predictions. This meets the
requirement that abstention reduce risk rather than merely suppress output.

V/A/D is explicitly marked unavailable. The existence of an architectural head
does not produce a quality claim without reviewed labels and calibration.

## 8. Final evaluation results

### 8.1 Main results

| Metric | Full precision | Mixed INT8 |
|---|---:|---:|
| Word error rate | 0.0734 | 0.0782 |
| Event frame macro-F1 | 0.6871 | 0.6877 |
| Localized-event segment macro-F1 | 0.7700 | 0.7772 |
| Event-presence macro-F1 | 0.8258 | 0.8220 |
| Style macro-F1 | 1.0000 | 1.0000 |
| Supported-class affect macro-F1 | 0.4746 | 0.4591 |
| Ontology-wide affect macro-F1 | 0.3560 | 0.3443 |
| OOD F1 | 0.9907 | 0.9747 |
| Affect coverage | 0.8258 | 0.8712 |
| Acoustic Preference Score | +0.2727 | +0.2727 |

All 20 executable release gates pass.

### 8.2 Semantic-acoustic reliance

The Acoustic Preference Score is positive for both final models. On the
semantic-conflict set, the model follows the acoustic target more often than
the conflicting lexical target. This provides evidence that audio contributes
information beyond transcript sentiment.

The conflict set remains small and acted, so this result does not establish
broad lexical-acoustic disentanglement across natural conversation.

### 8.3 Localized event results

| Event | Segment F1 | TP | FP | FN | Boundary MAE |
|---|---:|---:|---:|---:|---:|
| Laugh | 0.9195 | 40 | 2 | 5 | 156.8 ms |
| Cough | 0.6170 | 29 | 21 | 15 | 153.1 ms |
| Throat clear | 0.7733 | 29 | 5 | 12 | 127.2 ms |

Cough is the weakest localized event because it produces more insertions and
misses. Other event labels remain utterance-presence evidence only and are not
presented as temporally localized.

## 9. Quantization and export

### 9.1 Full-precision model

- File: `attune-split-tail-v0.1-fp.onnx`.
- Size: 968,740,218 bytes, approximately 924 MiB.
- Maximum PyTorch/ONNX absolute parity error: `8.39e-05`.
- Ten floating model outputs checked.

### 9.2 Mixed-INT8 model

- File: `attune-split-tail-v0.1-int8.onnx`.
- Size: 594,795,068 bytes, approximately 567 MiB.
- Size reduction: 38.6%.
- Model SHA-256:
  `d9b36f099aa456c6f1a1348099cb375d36b34433c898ef31fdbf7a24fe27e9aa`.

Most eligible shared linear weights use ONNX Runtime dynamic per-channel QInt8.
The upper perception blocks, frozen ASR tail, attentive pooling, and small
output heads remain full precision to meet the fixed quality gates. The release
is therefore accurately described as a mixed-precision INT8 graph rather than
a fully integer-only model.

## 10. Runtime and deployment

### 10.1 CPU benchmark

The final graph was benchmarked on the development Mac CPU using 40 balanced,
input-contract-valid clips representing 168.97 seconds of audio.

| Runtime measure | Result |
|---|---:|
| Real-time factor | 0.0534 |
| p50 latency | 206 ms |
| p95 latency | 415 ms |
| p99 latency | 440 ms |
| JSON validity | 100% |
| XML validity | 100% |
| Committed retractions | 0 |

One hundred 44.1 kHz DCASE source rows were explicitly excluded from this
runtime benchmark because the deployed input contract is 16 kHz mono PCM16.
They remained part of offline event evaluation.

### 10.2 Batch API

The FastAPI service exposes:

- `POST /v1/analyse`.
- `GET /healthz`.
- `GET /metrics`.

The real final backend returned HTTP 200, schema-v2 output, and runtime timing
metadata during the deployment smoke test.

### 10.3 Streaming API

The WebSocket service exposes `WS /v1/stream` and supports:

- `session.start`.
- `audio.chunk`.
- Provisional revisions.
- `audio.end`.
- One committed result.
- `session.close`.

The real-model smoke generated three provisional revisions followed by one
committed result. The server now rejects further audio after commitment,
enforcing the immutable-commit guarantee.

## 11. Software engineering completed

The repository now includes:

- Audited feature and source manifests.
- Corpus-aware joint data loading and batching.
- Multi-task masked loss functions.
- Frozen and upper-two adaptation policies.
- Atomic delta checkpoints and resume support.
- Frozen-head warm-start support.
- ONNX export with numerical parity validation.
- Selective mixed-precision quantization.
- Development calibration and fixed-threshold evaluation.
- ASR, event, style, affect, APS, OOD, calibration, and runtime metrics.
- Executable release gates.
- Batch and streaming inference services.
- A Python client and command-line inference tools.
- Deterministic artifact checksums.
- Dataset, model, ethics, schema, deployment, and annotation documentation.
- A technical report and final error analysis.

Final repository verification:

- 241 tests passed.
- 4 optional-model tests skipped.
- `ruff check .` passed.
- `ruff format --check .` passed across 198 Python files.

## 12. Major decisions and resolved failures

### Frozen affect probe rejected

The frozen-head candidate had strong event and OOD performance but failed the
affect gate. It was not promoted merely because it was convenient.

### STARSS23 path deprioritized

Earlier natural-scene STARSS23 experiments failed the predeclared temporal
boundary gate. Their negative results and error analyses were preserved, but
the failed path was not used to make unsupported localization claims.

### ASR degradation corrected architecturally

The first selected adapted model degraded evaluation-set ASR beyond the fixed
limit. Instead of weakening the gate, the model was restructured with a compact
frozen ASR tail. This preserved perception quality and restored base ASR.

### INT8 quality protected with selective precision

Fully dynamic INT8 and narrower mixed variants introduced a style-ranking
error. The fixed two-point parity gate was retained. Expanding the preserved FP
tail to the upper eight blocks restored style parity and retained acceptable
affect, event, OOD, WER, size, and runtime performance.

### Streaming commitment made truly immutable

Final hardening identified that the streaming server could accept more audio
after a committed result. The state machine was corrected and a regression test
was added.

## 13. Limitations and evidence boundaries

The following limitations must remain visible:

- Affect supervision is acted and one-hot, not naturalistic multi-rater gold.
- Only laugh, cough, and throat-clear have strong temporal supervision.
- Style evaluation is narrow and should not be interpreted as universal 100%
  performance.
- V/A/D is unavailable.
- `surprise` and `ambiguous` lack positive reviewed training classes.
- The semantic-conflict set is relatively small.
- Naturalistic cross-corpus affect generalization is not established.
- The final mixed model has not been rerun on the archived Ghanaian and British
  English slices.
- No human interaction study has demonstrated improved conversational
  responses.
- No permission to redistribute third-party-derived model weights is implied.

The initial evaluation partition was opened before the ASR split-tail correction.
The perception checkpoint and calibration remained fixed, but the corrected
ASR result should be confirmed on a new untouched external dataset before a
publication-level claim.

## 14. Work remaining for the broader research programme

The technical v0.1 model and deployment system are complete. The following are
future research stages:

1. Collect naturalistic, consented, multi-rater English affect data.
2. Add reviewed valence, arousal, and dominance supervision.
3. Create a new untouched external ASR and accent benchmark.
4. Expand Ghanaian, British, and other English-variety evaluation.
5. Run the preregistered downstream conversational human study.
6. Evaluate naturalistic cross-corpus affect and event performance.
7. Benchmark Linux GPU, commodity Linux CPU, and additional Apple hardware.
8. Complete legal review before releasing weights or audio.

These items are not silently represented as completed work in v0.1.

## 15. Principal deliverables

| Deliverable | Location |
|---|---|
| Mixed-INT8 model | `artifacts/models/attune-split-tail-v0.1-int8.onnx` |
| Full-precision model | `artifacts/models/attune-split-tail-v0.1-fp.onnx` |
| Selected trained delta | `artifacts/training/local-upper-two-v0.1/model.pt` |
| INT8 calibration | `artifacts/evaluation/split-tail-v0.1/int8-calibration.json` |
| FP calibration | `artifacts/evaluation/split-tail-v0.1/fp-calibration.json` |
| Final release gates | `artifacts/release/v0.1/release-gates.json` |
| Artifact checksum ledger | `artifacts/release/v0.1/artifact-manifest.json` |
| Deployment benchmark | `artifacts/deployment/split-tail-v0.1/int8-macos-cpu.json` |
| Service smoke report | `artifacts/deployment/split-tail-v0.1/int8-service-smoke.json` |
| Model card | `docs/model-card.md` |
| Dataset card | `docs/dataset-card-v0.1.md` |
| Technical report | `research/paper/attune-v0.1-technical-report.md` |
| Final error analysis | `research/error-analysis/v0.1-fp-sealed.md` |
| Usage guide | `README.md` |

## 16. Final position

Project Attune v0.1 demonstrates that a compact speech system can retain base
ASR, localize a limited set of vocal events, produce calibrated categorical
affect evidence, prefer acoustic delivery over conflicting text on the current
test set, abstain on uncertain cases, and run substantially faster than real
time on a local CPU.

The result is a functioning and reproducible paralinguistic transcription
layer. Its value lies in preserving audible evidence and uncertainty—not in
claiming that a machine can read a speaker's mind.
