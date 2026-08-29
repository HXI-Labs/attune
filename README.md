# Attune Cadence

Attune Cadence is an English speech model developed by Project Attune. It
combines a transcript with word timing, localized vocal events, perceived
affect probabilities, uncertainty, and abstention information. JSON schema v2
is the authoritative output; XML is rendered deterministically from validated
JSON.

The project describes audible expression, not a speaker's true emotional
state. Its output is not suitable for diagnosis, deception detection,
protected-trait inference, surveillance, or automated high-stakes decisions.

## Current status

The code, evaluation pipeline, local API, browser test interface, and deployment
exports are implemented. Model publication remains disabled.

A live hostile-speech test produced an accurate transcript but false
`whispering`, `cough`, and `sneeze` annotations. Runtime hardening now permits
only localized `laugh`, `cough`, and `throat_clear` events above 0.98
confidence. Utterance-level event and style allowlists are empty. The original
uploaded clip was processed in memory and was not retained, so a recording of
that case must be tested again before release.

ASR is not the current failure. On an external 480-clip RAVDESS inspection set,
the deployed INT8 lineage reached 0.0104 WER, while affect macro-F1 was 0.1213
and 70.8% of clips were classified as anger. The release gate therefore remains
closed while a frozen-encoder, full-head affect candidate is trained and
evaluated. The encoder and CTC parameters are locked during this work.

The last complete mixed-precision candidate had 241,609,098 parameters and a
595 MB ONNX graph. Its small reviewed sealed set produced 0.0782 WER, 0.6548
localized-event segment macro-F1, and no auxiliary false positives across 189
ordinary-speech controls. Those results do not establish broad naturalistic
emotion recognition. Detailed results and limitations are in
[`research/v0.1-implementation-status.md`](research/v0.1-implementation-status.md)
and [`research/ravdess-affect-external-v0.1.md`](research/ravdess-affect-external-v0.1.md).

## Output

For speech without supported paralinguistic evidence, a human-readable view may
be:

```text
[affect uncertain] I hate you, I hate you so much—never call me again.
```

The model returns structured data rather than generating the bracketed form:

```json
{
  "schema_version": "2.0",
  "transcript": {
    "text": "I hate you, I hate you so much, never call me again.",
    "confidence": 0.96,
    "words": [
      {"id": "w1", "text": "I", "start_ms": 80, "end_ms": 150, "confidence": 0.98}
    ]
  },
  "styles": [],
  "events": [],
  "affect": {
    "categories": {
      "neutral": 0.08,
      "joy": 0.01,
      "distress": 0.29,
      "anger": 0.38,
      "fear": 0.12,
      "surprise": 0.02,
      "other": 0.05,
      "ambiguous": 0.05
    },
    "top_label": null,
    "abstain": true,
    "abstention_reason": "No category passed the calibrated threshold."
  },
  "uncertainty": {
    "out_of_distribution_probability": 0.17,
    "interpretation_warning": "Vocal affect is a probabilistic perception, not a verified internal state."
  }
}
```

A supported localized event includes timestamps and confidence:

```json
{
  "label": "cough",
  "temporal_scope": "localized",
  "start_ms": 1450,
  "end_ms": 1680,
  "confidence": 0.99,
  "status": "committed"
}
```

Cadence v0.1 estimates one affect distribution per analysed utterance. If VAD or
turn boundaries separate a recording into several utterances, each utterance
can receive its own result. It does not yet claim affect-span tracking inside a
continuous utterance. Localized vocal events can still occur anywhere within
that utterance.

## Architecture

Cadence uses SenseVoice-Small as a shared acoustic encoder. The model has
separate heads for CTC transcription, frame-level events, event boundaries,
styles, categorical affect, valence/arousal/dominance, and out-of-distribution
scoring. Attentive statistics pooling produces the utterance-level affect
representation.

The deployment path keeps transcript content and model-produced metadata in
separate fields. Downstream applications should pass them through trusted
structured channels rather than concatenate markup into the spoken text.

The current export uses mixed precision. Most eligible shared weights are
dynamic per-channel INT8, while sensitive tail and output layers remain in
floating point. Export reports record the exact excluded nodes; the `INT8`
label does not mean that every operator is quantized.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev --extra torch --extra model-runners --extra deployment --extra serving
make check
```

`make check` verifies formatting, runs Ruff and pytest, and builds the source
distribution and wheel.

Installation does not download model checkpoints. Review the SenseVoice model
agreement before setting `ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1`. Local weights
are expected below `data/raw/`, which is ignored by Git.

## Inference and local service

The command-line runner accepts PCM16, 16 kHz, mono WAV files between 0.5 and
30 seconds:

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/infer_onnx.py \
  --model artifacts/models/attune-split-tail-v0.1-int8.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/split-tail-v0.1/int8-calibration.json \
  --quantization int8 \
  --xml input.wav
```

Run the same backend behind FastAPI:

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/serve.py \
  --model artifacts/models/attune-split-tail-v0.1-int8.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/split-tail-v0.1/int8-calibration.json \
  --quantization int8
```

The service exposes `POST /v1/analyse`, `WS /v1/stream`, `/healthz`, and
Prometheus-compatible `/metrics`. The root route serves a small test interface
with upload, microphone recording, audio normalization, structured evidence,
and raw JSON. Batch uploads are capped at 2 MiB; valid PCM16 input within the
30-second model limit is smaller than that cap.

Every route must be authenticated before the service is exposed through a
public tunnel. Temporary Basic authentication is configured with environment
variables:

```bash
ATTUNE_DEMO_USERNAME=attune-test \
ATTUNE_DEMO_PASSWORD='generate-a-new-secret' \
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 \
uv run python scripts/serve.py \
  --model artifacts/models/attune-split-tail-v0.1-int8.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/split-tail-v0.1/int8-calibration.json \
  --quantization int8
```

Uploads are processed in memory. A reverse proxy or tunnel still handles audio
in transit and requires its own security review.

## Training and evaluation

Training uses checksum-verified JSONL manifests and delta checkpoints. Missing
task labels are masked rather than interpreted as negatives. Each run records
the manifest hash, training-source hash, seed, configuration, parameter counts,
loss history, elapsed time, and estimated compute cost.

The current frozen full-head protocol is documented in
[`research/frozen-full-head-v0.6-protocol.md`](research/frozen-full-head-v0.6-protocol.md).
The wider reproduction guide is
[`docs/cloud-training.md`](docs/cloud-training.md). Dataset and weight terms are
recorded under [`data/provenance/`](data/provenance/); third-party audio and
weights are not redistributed.

Important repository areas are:

- `src/attune/models/`: shared model and research heads;
- `src/attune/training/`: manifests, losses, batching, checkpoints, and audits;
- `src/attune/evaluation/`: ASR, event, affect, calibration, and release metrics;
- `src/attune/schema/`: schema v1/v2 models and deterministic XML rendering;
- `src/attune/inference/`: ONNX inference, streaming, export, and quantization;
- `src/attune/service/`: FastAPI routes, authentication, and browser interface;
- `scripts/`: reproducible data, training, evaluation, and deployment commands;
- `research/`: experiment registry, results, and error analysis.

## Licence and use restrictions

Project code is MIT licensed. Dataset and model licences remain independent.
SenseVoice-Small weights use the
[FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).
Redistribution of derived weights requires a separate review of base-model and
training-data terms.

Do not use Attune for covert monitoring, diagnosis, deception detection,
protected-trait inference, or automated hiring, credit, insurance, policing,
medical, legal, or other consequential decisions. Uncertain output should lead
to cautious clarification, not an asserted emotion.
