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

Cadence v0.1 is an experimental research preview implemented in full precision
and INT8. The source code is suitable for publication and review. The model
weights are uploaded privately first because the recorded SenseVoice
redistribution review does not yet approve a public derivative-weight release.

The preview contains the 241,904,650-parameter Cadence graph and a 30,726-parameter
calibrated event head. The combined deployment remains below 242 million
parameters. The INT8 graph is 500 MB, 48.4% smaller than the 970 MB
full-precision graph.

The release candidate supports ASR, word timing, affect distributions,
abstention, and conservative vocal-event output. Vocal styles are disabled.
A four-way sound-event style head hallucinated `whispering` on ordinary speech,
and a replacement BERSt shouting head missed its fixed recall and F1 gates.
Both experiments were rejected rather than hidden behind a higher runtime
threshold.

The event head reaches macro-F1 0.814 through the full-precision ONNX graph and
0.826 through INT8 on the 80-clip opened VocalSound inspection set. Both graphs
produce two false event emissions across 160 external OOD speech and sound
controls, a 1.25% false-positive clip rate. These are source-labelled opened
benchmarks, not a claim of natural inline-event accuracy.

Four neutral system voices reading `I hate you, I hate you so much, never call
me again` are the exact regression for the failure that prompted this release.
Both FP32 and INT8 transcribe all four correctly, return no events or styles,
and abstain on affect. The INT8 FastAPI and pseudo-streaming smoke test commits
the same transcript at real-time factor 0.055 on the development Mac.

The retained affect model reaches macro-F1 0.6643 on development, 0.6336 on the
opened regression set, and 0.3853 on external RAVDESS. INT8 reaches 0.3803 on
the same RAVDESS set, a 0.51-point absolute loss. Both external results remain
below the project's 0.40 public-release target. A fresh consented human
recording of the hostile-speech regression and the derivative-weight licensing
review also remain required before public model publication. The current
artifact is therefore a research preview, not a validated public emotion
model. An internal joy-to-distress transition regression also exposed a
specific compact-head failure: both source clips were classified as fear even
though the emotion2vec+ teacher distinguished them correctly. The locked
composition and acceptance checks are in
[`research/release-cascade-v0.1-protocol.md`](research/release-cascade-v0.1-protocol.md).

The next accuracy candidate adds a truncated three-block emotion2vec+ branch
and remains below 300 million active parameters. The v0.13 head uses a small
probability-replay update over v0.11 to reduce unsupported named-affect output
on ordinary speech. Fixed Cadence fusion reaches macro-F1 0.6401 on paired
CREMA-D development, 0.3070 on BERSt development, and 0.8322 on external
RAVDESS. On 100 untouched British Common Voice speakers, named affect is the
top category for 3 clips and only 1 clears the default 0.40 confidence gate.
On the untouched joy-to-distress composite, it emits separate joy and distress
spans and abstains on the mixed utterance as a whole. This candidate is not the
published v0.1 model; its method and remaining release work are documented in
[`research/affect-control-finetune-v0.13-results.md`](research/affect-control-finetune-v0.13-results.md).

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

Cadence v0.1 returns an utterance-level affect distribution and experimental
non-overlapping affect windows for recordings longer than four seconds. These
windows make changing delivery visible in the interface, but they are fixed
analysis windows rather than learned transition boundaries. Their categorical
accuracy has not passed the release gate. Localized vocal events can still
occur anywhere within the utterance.

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

To run the unreleased v0.13 affect candidate, add its portable INT8 acoustic
branch. The runner then uses the fixed 0.86 acoustic fusion weight and 0.40
confidence threshold unless explicitly overridden:

```bash
  --affect-torchscript artifacts/models/cadence-affect-student-v0.13/affect-int8.pt \
  --affect-calibration artifacts/models/cadence-affect-student-v0.13/fused-calibration.json
```

The 81.8 MB TorchScript artifact contains the truncated emotion2vec+ branch and
its dynamic INT8 linear layers. It accepts both waveform and padding-mask
inputs, so single clips and mixed-length batches use the same graph. The
calibration file is bound to the model SHA-256 and is rejected if paired with a
different artifact.

Speaker-grouped five-fold evaluation on 607 development clips improved INT8
macro-F1 from 0.3544 to 0.3702 after calibration, compared with 0.3766 for the
FP32 reference. On the separate 480-clip RAVDESS evaluation, the calibrated
artifact reaches 0.8694 fused macro-F1 and a 0.072 affect-branch real-time
factor on the development Mac. The uncalibrated artifact reaches 0.8311, and
the FP32 reference reaches 0.8322.

A padding-aware ONNX export is retained for research comparisons. Its 0.3493
development macro-F1 misses the two-point FP32 parity gate, so it is not the
default portable path.

The PyTorch runner remains available for training comparisons:

```bash
  --emotion2vec-path data/raw/model-cache/emotion2vec-plus \
  --affect-student artifacts/training/truncated-emotion2vec-affect-control-finetune-v0.13/head.pt \
  --affect-quantization int8
```

This mode dynamically quantizes supported linear layers at load time and is not
the portable release path.

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
  --model artifacts/models/attune-cadence-v0.1-int8.onnx \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --calibration artifacts/evaluation/affect-focus-v0.9/calibration.json \
  --probe-head artifacts/release-candidate/vocalsound-en-head.npz \
  --quantization int8
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
  --repo-id jbuaba/attune-cadence-242m \
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
  --probe-head artifacts/release-candidate/vocalsound-en-head.npz \
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
