# Attune Cadence

Attune Cadence is a compact English speech-perception system that preserves
both what was said and how it sounded. For a 0.5–30 second voice recording, it
returns a transcript, word timing, localized vocal events, phrase-level
perceived affect, confidence, and abstention in one time-aligned JSON result.

Cadence reports observable vocal evidence and uncertain listener perception;
it does not claim to determine a speaker's internal emotional state. The JSON
schema is authoritative. XML and bracketed transcripts are deterministic views
generated from the validated result.

## Cadence v0.13

v0.13 is the current evaluation model. It keeps the 241,904,650-parameter
Cadence INT8 graph for transcription, timing, and vocal events, and adds a
57,937,364-parameter affect branch. The complete model uses **299,842,014 active
parameters**.

The model returns transcripts, word timing, conservative vocal events,
phrase-level perceived affect, and abstention. Vocal styles and dimensional
valence, arousal, and dominance remain disabled until they pass the project's
evaluation gates.

The [model package](https://huggingface.co/jbuaba/attune-cadence/tree/main/candidate-v0.13)
is currently private. v0.13 is intended for consented research and evaluation,
not as a verified emotion detector or for automated high-stakes decisions.

Detailed training, calibration, quantization, accuracy, controls, and known
limitations are recorded in the [v0.13 results](research/affect-control-finetune-v0.13-results.md).

## Output

The structured result can be rendered as a compact readable transcript. These
examples illustrate the intended Attune experience; labels that have not yet
passed the release gates remain disabled in the current checkpoint.

```text
[laughing speech; perceived joy 81%]
I cannot believe you actually did that! [laugh]

[whispering; perceived fear 63%]
Did you hear that outside? [breath]

[crying speech; perceived distress 76%]
I said I was fine. [sob]

[shouting; perceived anger 72%]
Put my cake back in the fridge!
```

The same words can produce different results when their audible delivery
changes:

```text
[neutral] I'm fine.
[laughing speech; perceived joy 78%] I'm fine. [laugh]
[crying speech; perceived distress 69%] I'm fine. [sob]
[affect uncertain] I'm fine.
```

When turn or VAD boundaries separate changing delivery into distinct
utterances, a conversation can read:

```text
[neutral] I thought the parcel was lost.
[laughing speech; perceived joy 74%] It was behind the door the whole time. [laugh]
[crying speech; perceived distress 66%] I really needed that today. [sob]
```

Square brackets are a human-readable projection, not model-generated markup.
JSON remains authoritative and keeps transcript text separate from trusted
model metadata.

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

The current runtime returns an utterance-level affect distribution and
experimental non-overlapping affect windows for recordings longer than four
seconds. These are fixed analysis windows, not learned transition boundaries,
and have not passed the release gate. Localized vocal events can still occur
anywhere within the utterance.

## Architecture

Cadence uses SenseVoice-Small as a shared acoustic encoder. One ONNX pass
produces CTC transcription, frame-level event outputs, affect outputs, and a
padding-safe 5,120-value acoustic embedding. A small NumPy linear head consumes
that embedding for calibrated utterance-level vocal events. The head does not
require PyTorch at inference time. Raw SenseVoice event and emotion tags are
not used.

The deployment path keeps transcript content and model-produced metadata in
separate fields. Downstream applications should pass them through trusted
structured channels rather than concatenate markup into the spoken text.

The release artifacts are `attune-cadence-v0.1-fp.onnx`,
`attune-cadence-v0.1-int8.onnx`, and `vocalsound-en-head.npz`. Styles remain in
the research architecture but have no enabled release labels.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev --extra model-runners --extra dataset-tools
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
  --model artifacts/models/attune-cadence-v0.1-int8.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/affect-focus-v0.9/calibration.json \
  --probe-head artifacts/release-candidate/vocalsound-en-head.npz \
  --quantization int8 \
  --xml input.wav
```

To run the v0.13 affect candidate, add its portable INT8 acoustic branch:

```bash
  --affect-torchscript artifacts/models/cadence-affect-student-v0.13/affect-int8.pt \
  --affect-calibration artifacts/models/cadence-affect-student-v0.13/fused-calibration.json
```

The calibration is bound to the model SHA-256 and is rejected when paired with
a different artifact. Full accuracy, parity, and runtime measurements are in
the [v0.13 results](research/affect-control-finetune-v0.13-results.md).

Run the same backend behind FastAPI:

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/serve.py \
  --model artifacts/models/attune-cadence-v0.1-int8.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/affect-focus-v0.9/calibration.json \
  --probe-head artifacts/release-candidate/vocalsound-en-head.npz \
  --quantization int8
```

The service exposes `POST /v1/analyse`, `WS /v1/stream`, `/healthz`, and
Prometheus-compatible `/metrics`. The root route provides microphone recording,
file upload, structured evidence, and raw JSON. Batch uploads are processed in
memory and capped at 2 MiB. Run `scripts/serve.py --help` for optional network
and Basic Authentication settings.

## Training and evaluation

Training uses checksum-verified JSONL manifests and delta checkpoints. Missing
task labels are masked rather than interpreted as negatives. Each run records
the manifest hash, training-source hash, seed, configuration, parameter counts,
loss history, elapsed time, and estimated compute cost.

The current affect protocol and results are in
[`research/affect-control-finetune-v0.13-results.md`](research/affect-control-finetune-v0.13-results.md).
The [training guide](docs/cloud-training.md) covers local and rented-GPU runs.
Dataset and weight terms are recorded under [`data/provenance/`](data/provenance/);
third-party audio and weights are not redistributed.

## Documentation

- [Output schema](docs/schema-v2.md)
- [Ontology](docs/ontology.md)
- [Dataset card](docs/dataset-card-v0.1.md)
- [Model card](docs/model-card.md)
- [Annotation guide](docs/annotation-guide.md)
- [Ethics and use constraints](docs/ethics.md)
- [Release checklist](docs/release-checklist.md)

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
