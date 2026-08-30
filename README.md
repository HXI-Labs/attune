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

The code, evaluation pipeline, local API, browser test interface, and ONNX
export path are implemented. Model publication remains disabled.

A live hostile-speech test produced an accurate transcript but false
`whispering`, `cough`, and `sneeze` annotations. Runtime hardening now permits
only localized `laugh`, `cough`, and `throat_clear` events above 0.98
confidence. Utterance-level event and style allowlists are empty. The original
uploaded clip was processed in memory and was not retained, so a recording of
that case must be tested again before release.

As a lexical-leakage control, four neutral system voices subsequently read the
exact hostile sentence. Cadence transcribed all four correctly, returned no
events or styles, and abstained on affect; anger probability ranged from 8.1%
to 19.9%. This establishes that the words alone do not reproduce the failure,
but synthetic neutral speech cannot validate expressive human delivery. The
protocol and checksums are recorded in
[`research/hostile-lexical-control-v0.1.md`](research/hostile-lexical-control-v0.1.md).

ASR is not the current failure. Cadence v0.9 keeps a frozen base-ASR tail; its
CTC logits are bit-identical to the preceding candidate on real speech clips.
The 241,904,650-parameter model adapts the upper acoustic encoder for perception
and then trains isolated event, style, and affect branches. Its self-contained
delta contains 7,905,483 learned parameters; the remaining base weights stay
frozen.

The current candidate reaches affect macro-F1 of 0.6643 on development, 0.6336
on the opened regression set, and 0.3853 on the external 480-clip RAVDESS set.
Localized-event segment macro-F1 is 0.6204 on opened regression data, with no
localized false positives across 549 regression speech controls or 480
RAVDESS speech clips. External WESR results remain weak: temporal event
presence macro-F1 is 0.2882 and style macro-F1 is 0.3795. Event-presence and
style allowlists therefore remain empty, quantization is deferred, and the
fresh confirmation set remains sealed. Protocols and complete results are in
[`research/affect-focus-v0.9-protocol.md`](research/affect-focus-v0.9-protocol.md),
[`research/event-hardening-v0.8-protocol.md`](research/event-hardening-v0.8-protocol.md),
and [`research/style-branch-v0.7-protocol.md`](research/style-branch-v0.7-protocol.md).

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

The active v0.9 candidate is a full-precision ONNX export. INT8 work is deferred
until the external affect, event, and style gates pass; quantizing an unreleasable
candidate would not resolve its data-generalization failures.

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
  --model artifacts/models/attune-affect-focus-v0.9-fp.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/affect-focus-v0.9/calibration.json \
  --quantization fp32 \
  --xml input.wav
```

Run the same backend behind FastAPI:

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/serve.py \
  --model artifacts/models/attune-affect-focus-v0.9-fp.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/affect-focus-v0.9/calibration.json \
  --quantization fp32
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
  --model artifacts/models/attune-affect-focus-v0.9-fp.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/affect-focus-v0.9/calibration.json \
  --quantization fp32
```

Uploads are processed in memory. A reverse proxy or tunnel still handles audio
in transit and requires its own security review.

## Training and evaluation

Training uses checksum-verified JSONL manifests and delta checkpoints. Missing
task labels are masked rather than interpreted as negatives. Each run records
the manifest hash, training-source hash, seed, configuration, parameter counts,
loss history, elapsed time, and estimated compute cost.

The current affect-focused protocol is documented in
[`research/affect-focus-v0.9-protocol.md`](research/affect-focus-v0.9-protocol.md).
The wider reproduction guide is
[`docs/cloud-training.md`](docs/cloud-training.md). Dataset and weight terms are
recorded under [`data/provenance/`](data/provenance/); third-party audio and
weights are not redistributed.
The complete model-weight release sequence is
[`docs/release-checklist.md`](docs/release-checklist.md).

Important repository areas are:

- `src/attune/models/`: shared model and research heads;
- `src/attune/training/`: manifests, losses, batching, checkpoints, and audits;
- `src/attune/evaluation/`: ASR, event, affect, calibration, and release metrics;
- `src/attune/schema/`: schema v1/v2 models and deterministic XML rendering;
- `src/attune/inference/`: ONNX inference, streaming, export, and quantization;
- `src/attune/service/`: FastAPI routes, authentication, and browser interface;
- `scripts/`: reproducible data, training, evaluation, and deployment commands;
- `research/`: experiment registry, results, and error analysis.

## Release publishing

Hugging Face publication is manifest-driven. A release bundle manifest names
the gate report and every local source-to-repository file mapping with its
SHA-256 digest. The publisher rejects missing or changed files, incomplete or
unsafe destinations, stale gate schemas, failed external event/style/affect
validation, and a missing or failed human hostile-speech regression.
External event, style, and affect gates are rerun independently after INT8
quantization and recalibration; full-precision acceptance is not reused.

After a candidate passes the full-precision and INT8 release suite, inspect the
upload plan before publishing:

```bash
uv run python scripts/publish_huggingface.py \
  --repo-id buabaj/attune-cadence-241m \
  --bundle-manifest artifacts/release/cadence-v0.1/huggingface-bundle.json \
  --dry-run
```

Create the checksum ledger and `huggingface-bundle.json` from the committed
release inventories:

```bash
uv run python scripts/build_release_manifest.py \
  --artifact-list configs/release/v0.1-artifacts.txt \
  --output artifacts/release/cadence-v0.1/artifact-manifest.json

uv run python scripts/build_huggingface_bundle.py \
  --release-name "Attune Cadence v0.1" \
  --gate-report artifacts/release/cadence-v0.1/release-gates.json \
  --file-list configs/release/v0.1-huggingface-files.txt \
  --output artifacts/release/cadence-v0.1/huggingface-bundle.json
```

Both builders fail if a listed artifact is absent. They hash every source, and
the publisher recalculates those hashes immediately before upload.

The hostile-speech gate cannot be enabled with a bare Boolean flag. Analyse a
fresh, consented, event-free human recording of the regression sentence with
the final graph, then create the hashed evidence report:

```bash
uv run python scripts/infer_onnx.py hostile-human.wav \
  --model <final-model.onnx> \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration <final-calibration.json> \
  --quantization int8 \
  --output-dir artifacts/release/cadence-v0.1/hostile

uv run python scripts/check_hostile_speech_regression.py \
  --audio hostile-human.wav \
  --inference artifacts/release/cadence-v0.1/hostile/hostile-human.attune.json \
  --model <final-model.onnx> \
  --calibration <final-calibration.json> \
  --human-recording-confirmed \
  --speaker-consent-confirmed \
  --output artifacts/release/cadence-v0.1/hostile-speech-regression.json
```

The check requires an exact normalized transcript at confidence 0.90 or above
and no event or style output for that deliberately event-free, ordinary-voice
recording. The release bundle includes the report, not the identifiable WAV.

Omit `--dry-run` to upload privately. Public repository creation additionally
requires `--public` plus `--redistribution-review`. The committed review record
currently denies public weight redistribution. Public upload requires an
approval covering the pinned SenseVoice revision, the exact training-source
set, and the SHA-256 digest of every ONNX artifact in the bundle.
Private upload does not override the upstream model agreement or dataset terms.

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
