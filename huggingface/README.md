---
language:
  - en
license: other
license_name: funasr-model-open-source-license-agreement-v1.1
license_link: https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
base_model: FunAudioLLM/SenseVoiceSmall
pipeline_tag: automatic-speech-recognition
library_name: onnxruntime
tags:
  - audio
  - speech
  - automatic-speech-recognition
  - paralinguistics
  - affective-computing
  - sound-event-detection
  - onnx
  - int8
---

# Attune Cadence 242M

Attune Cadence is a compact English paralinguistic transcription model. It
returns CTC transcription and word timing together with conservative vocal
events, perceived-affect probabilities, OOD information, and calibrated
abstention. Schema-v2 JSON is authoritative; XML and bracketed text are
deterministic views.

This package is a private research release candidate. Do not make it public
until the included release gates, human regression, affect confirmation, and
weight-redistribution review all pass.

## Scope

- English, single-speaker clips or user turns.
- 16 kHz mono PCM16 WAV input, 0.5–30 seconds.
- CTC transcript and approximate word timings.
- Calibrated `laugh`, `sigh`, `cough`, `throat_clear`, and `sneeze` events.
- One perceived-affect distribution per analysed utterance.
- Abstention and OOD probability.
- Vocal styles and V/A/D disabled in v0.1.

For ordinary speech without supported paralinguistic evidence, a readable view
can be:

```text
[affect uncertain] I hate you, I hate you so much—never call me again.
```

Four neutral system voices reading that exact sentence are the regression for
the false-tag failure that prompted this candidate. FP32 and INT8 transcribe
all four correctly, produce no events or styles, and abstain on affect.

## Architecture and files

The SenseVoice-derived ONNX graph has 241,904,650 parameters. A separate
30,726-parameter NumPy event head consumes a padding-safe acoustic embedding
from the same graph. The unique total is 241,935,376 parameters.

- `attune-cadence-242m-int8.onnx` — recommended 500 MB deployment graph.
- `attune-cadence-242m-fp.onnx` — 970 MB full-precision reference.
- `event-head.npz` — calibrated event head; no PyTorch dependency.
- `calibration.json` — affect and OOD calibration.
- `attune-output-v2.0.schema.json` — authoritative output contract.
- `evidence/` — evaluation and deployment reports.
- `release-gates.json` — executable publication decision.

The Python runtime also needs the official SenseVoiceSmall frontend and decoder
assets obtained separately under the upstream agreement.

## Measured results

| Metric | Full precision | INT8 |
|---|---:|---:|
| VocalSound event macro-F1, 80 opened clips | 0.814 | 0.826 |
| External OOD false-positive rate, 160 clips | 0.0125 | 0.0125 |
| Exact hostile-lexical transcripts | 4/4 | 4/4 |
| False events/styles on those controls | 0 | 0 |

The retained full-precision affect graph reaches macro-F1 0.6643 on
development, 0.6336 on an opened regression set, and 0.3853 on external
RAVDESS. INT8 reaches 0.3803 on the same 480 clips, an absolute loss of 0.0051.
Both external results are below the project's 0.40 public-release target. No
public affect-generalization claim is supported.

The event sets are source-labelled, partly isolated sounds rather than reviewed
natural inline events. The four lexical controls are synthetic neutral speech.
These opened results do not replace a sealed human evaluation.

## Intended use and safety

Use Cadence only as uncertain supplementary evidence in consented speech
research, accessibility experiments, and low-stakes conversational interfaces.
Do not use it for covert monitoring, diagnosis, deception detection, protected-
trait inference, speaker identification, or automated high-stakes decisions.
Do not state that a speaker *is* an emotion based on this output.

No controlled downstream human study has established that this representation
improves conversational responses.

## Licence and attribution

Cadence is derived from **SenseVoiceSmall by FunASR/FunAudioLLM**. Its weights
and derivatives are governed by the
[FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE),
separately from the MIT-licensed Attune source code. Dataset terms and the
current redistribution decision are included with the project. A private
Hugging Face upload is not public-release clearance.
