# Attune Cadence

**Hear how it was said.**

Attune Cadence is the first model from Project Attune: a compact, calibrated
paralinguistic transcription layer for English voice interaction. It returns
what was said, where supported vocal events occurred, how the delivery sounded,
and how uncertain the interpretation is. The authoritative output is schema-v2
JSON; XML is a deterministic view of that validated data.

Attune does **not** infer a speaker's true internal state. Its affect output is a
fallible perception distribution, not a diagnosis, deception judgment, safety
decision, or protected-trait inference.

## v0.1 accuracy-hardening status

The local 241,609,098-parameter SenseVoice-Small derivative is **not currently
release-ready**. A live test exposed severe false-positive `whispering`,
`cough`, and `sneeze` tags despite an accurate transcript. The public demo,
release PR, and Hugging Face publication were paused. The release gate now
fails closed until the same hostile-speech audio is retested successfully.

The hardened candidate keeps the accurate CTC route unchanged and narrows the
user-facing auxiliary output:

- the lower encoder is shared;
- only temporally localized `laugh`, `cough`, and `throat_clear` spans may be
  emitted, and only at confidence >= 0.98;
- weak utterance event-presence and style outputs are disabled because their
  isolated-sound evaluation did not validate them on ordinary speech;
- categorical affect remains probabilistic and may abstain;
- frozen copies of only the two original upper encoder blocks preserve CTC ASR;
- V/A/D is unavailable because the reviewed training bundle has no dimensional
  labels; and
- real-model regression checks preserve the transcript while requiring no
  unsupported auxiliary tags on ordinary speech.

The deployment graph is mixed precision: most eligible shared weights are
dynamic per-channel INT8, while the two high-level perception/ASR tails and
small output heads remain FP. It is 595 MB versus 969 MB for FP (38.6% smaller).
Calling it “INT8” does not imply every operator is quantized; the exact excluded
nodes are recorded in its quantization report.

### Sealed evaluation

| Metric | FP | mixed INT8 |
|---|---:|---:|
| WER | 0.0734 | 0.0782 |
| Localized-event segment macro-F1 at >= 0.98 | 0.6645 | 0.6548 |
| Ordinary-speech auxiliary false-positive rate (189 clips) | 0.0000 | 0.0000 |
| Ordinary-speech localized false events/minute | 0.0000 | 0.0000 |
| Event-presence macro-F1 | 0.8258 | 0.8220 |
| Style macro-F1 | 1.0000 | 1.0000 |
| Supported-class affect macro-F1 | 0.4746 | 0.4591 |
| OOD F1 | 0.9907 | 0.9747 |
| Affect coverage | 0.8258 | 0.8712 |
| Acoustic Preference Score | +0.2727 | +0.2727 |

Event-presence and style scores above are retained as research diagnostics;
those heads are not emitted by the hardened runtime. The style result comes
from isolated weak-label audio and does not demonstrate reliable
speech-embedded shouting or whispering. The FP selective affect error is 0.4128
versus 0.4773 at full coverage. These are engineering results on a small,
source-labelled, partly acted/synthetic bundle—not evidence of broad
naturalistic emotion understanding.

### Negative-control follow-up

An opt-in v0.2 experiment now distinguishes explicit weak speech negatives from
missing labels. A lightweight correction over the frozen INT8 embeddings
reached 0.8152 sealed event-presence macro-F1, but still labeled one neutral
CREMA-D speech control as `sneeze` at 0.9976. It is therefore **not deployed**,
and style/event-presence runtime allowlists remain empty. See
`research/auxiliary-negative-controls-v0.2.md` for the complete train/dev/sealed
protocol and per-label results.

The first sealed pass exposed excessive ASR drift in the originally selected
single-tail model. The perception checkpoint and calibration were kept fixed,
and the architecture was corrected with the frozen ASR tail described above.
The corrected graph was then evaluated on the same sealed set. Consequently,
the perception result retains its original sealed status, while the corrected
ASR result should be confirmed on a new external untouched set before a paper
claim.

### Runtime

On the development Mac CPU, the final mixed-INT8 graph processed 40 valid
16 kHz mono clips at 0.0534 real-time factor, with p50/p95/p99 latency of
206/415/440 ms, 100% valid JSON and XML, and zero committed-output retractions.
A real FastAPI/WebSocket smoke emitted three provisional revisions followed by
one committed result. Hardware-specific results are in
`artifacts/deployment/split-tail-v0.1/`.

No downstream human-response study has been run, so v0.1 does **not** claim that
Attune improves conversational responses. That study is specified in
`docs/human-interaction-roadmap.md`.

## What a transcription looks like

For the hostile sentence that exposed the false-positive failure, the hardened
policy should return the accurate transcript without inventing weak evidence:

```text
[affect uncertain] I hate you, I hate you so much—never call me again.
```

The original uploaded audio was processed in memory and was not retained, so
this exact case remains a mandatory manual retest—not a claimed passing result.
For a genuine cough whose localized confidence exceeds 0.98, a shortened XML
excerpt can be:

```xml
<attune schema_version="2.0">
  <transcript confidence="0.93">
    <text>I said leave me alone.</text>
    <words>...</words>
  </transcript>
  <styles />
  <events>
    <event id="event-1" label="cough" temporal_scope="localized"
           start_ms="1450" end_ms="1680" confidence="0.99"
           status="committed" />
  </events>
  <affect start_ms="0" end_ms="1800" abstain="false">...</affect>
</attune>
```

The bracketed transcription is a human-readable illustration, not a second
model-generated format. The model actually returns
validated JSON containing the transcript, any supported localized event timing,
all affect probabilities, abstention state, OOD probability, and an
interpretation warning. Word timestamps are grouped from genuine CTC token
spans using the tokenizer's SentencePiece boundaries; no uniform timing is
fabricated.

### Different delivery within one recording

The long-term intended experience for separately detected speech segments is:

```text
[neutral] I’ll take care of it.
[shouting; perceived anger 76%] But don’t ask me again!
[laughing speech; perceived joy 64%] I’m only joking. [laugh]
```

Cadence v0.1 does not currently emit the style labels shown in that target
experience. It produces one affect distribution per analysed utterance. When VAD
or turn boundaries separate these lines, each segment can receive its own
result. If the speaker changes delivery without a usable boundary inside one
continuous utterance, v0.1 may return a mixed distribution or abstain:

```text
[shouting detected; affect ambiguous]
I’ll take care of it, but don’t ask me again! I’m only joking. [laugh]
```

True within-utterance affect-span tracking is not claimed in v0.1. Supported
localized events can still carry timestamps; affect remains utterance-scope.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev --extra torch --extra model-runners --extra deployment --extra serving
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Installation never downloads model checkpoints. Review the official
SenseVoice model agreement before setting the acknowledgement variable.

## Local inference

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/infer_onnx.py \
  --model artifacts/models/attune-split-tail-v0.1-int8.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/split-tail-v0.1/int8-calibration.json \
  --quantization int8 --xml input.wav
```

Inputs must be RIFF/WAV, PCM16, 16 kHz, mono, and no longer than 30 seconds.
The CLI emits validated schema-v2 JSON and optional deterministic XML.

Serve the same backend locally:

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/serve.py \
  --model artifacts/models/attune-split-tail-v0.1-int8.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/split-tail-v0.1/int8-calibration.json \
  --quantization int8
```

The service exposes `POST /v1/analyse`, `WS /v1/stream`, `/healthz`, and
Prometheus-compatible `/metrics`. Its `/` route provides the Cadence test
interface with file upload, in-browser audio conversion, microphone recording,
compact evidence rendering, and raw JSON. Spoken transcript content and
model-produced metadata remain separate trusted fields.

Any internet-facing demo must protect every route. Temporary Basic Auth can be
enabled without storing credentials in the repository:

```bash
ATTUNE_DEMO_USERNAME=cadence-test \
ATTUNE_DEMO_PASSWORD='generate-a-new-secret' \
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 \
uv run python scripts/serve.py \
  --model artifacts/models/attune-split-tail-v0.1-int8.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/split-tail-v0.1/int8-calibration.json \
  --quantization int8
```

This protects HTTP and WebSocket routes. Audio is processed in memory; the demo
does not persist uploads. A reverse proxy or tunnel still handles audio in
transit and must be reviewed separately.

## Reproducibility and evidence

- Release gates (currently `release_ready: false`):
  `artifacts/release/v0.1/release-gates.json`
- Artifact hashes: `artifacts/release/v0.1/artifact-manifest.json`
- Training run: `artifacts/training/local-upper-two-v0.1/`
- Development/sealed reports: `artifacts/evaluation/split-tail-v0.1/`
- Runtime reports: `artifacts/deployment/split-tail-v0.1/`
- Model card: `docs/model-card.md`
- Dataset card: `docs/dataset-card-v0.1.md`
- Implementation report: `research/v0.1-implementation-status.md`
- Training/reproduction guide: `docs/cloud-training.md`

The checksum-verified training manifest contains 2,490 rows / 3.79 hours:
1,691 train, 398 development, and 401 sealed test. Missing task labels are
masked rather than treated as negatives. Audio and third-party model weights
are not redistributed by this repository.

## Repository map

- `src/attune/models/joint.py` — shared encoder, split ASR tail, and task heads
- `src/attune/training/` — audited manifests, objectives, batching, checkpoints
- `src/attune/evaluation/` — ASR/event/affect/OOD/deployment metrics and gates
- `src/attune/schema/` — authoritative v2 models and safe XML renderer
- `src/attune/inference/` — ONNX backend, streaming state, export, quantization
- `src/attune/service/` — batch and pseudo-streaming FastAPI service
- `scripts/` — reproducible preparation, training, evaluation, and serving CLIs
- `docs/` — ontology, ethics, model/data cards, protocols, and deployment guide
- `research/` — experiment registry, historical baselines, and error analyses

## Licence and safety boundary

Project code is MIT licensed. Dataset and model licences remain independent.
Official SenseVoice-Small weights use the
[FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)
and require their own attribution/release review. Exact dated records are in
`data/provenance/`.

Do not use Attune for covert monitoring, diagnosis, deception detection,
protected-trait inference, or automated hiring, credit, insurance, policing,
medical, legal, or other high-stakes decisions. Uncertain output should prompt
cautious clarification, not an asserted emotion.

## Model identity

- Public name: **Attune Cadence**
- Release ID: **`attune-cadence-241m`**
- Base model attribution: **SenseVoiceSmall by FunASR/FunAudioLLM**
- Parameter count: **241,609,098**
- Deployment variants: full-precision ONNX and mixed-precision INT8 ONNX
