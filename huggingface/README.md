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

# Attune Cadence 241M

Attune Cadence is an English paralinguistic transcription model from
HXI Labs. It combines CTC speech transcription with supported vocal-event
localization, perceived-affect probabilities,
out-of-distribution detection, calibrated abstention, and word timing.

Publication is blocked. A live test exposed false-positive event and style
tags. The hardened runtime disables weak event-presence and style output,
requires at least 0.98 confidence for localized events, and still requires a
manual hostile-speech regression. A later external RAVDESS evaluation also
found only 0.1213 affect macro-F1 and a 70.8% anger prediction share. Do not
publish this package yet.

This v0.1 checkpoint is a 241,609,098-parameter derivative of
[SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) by
FunASR/FunAudioLLM. The uploaded deployment graph is mixed precision: most
eligible shared linear weights are QInt8, while accuracy-sensitive upper blocks
and small heads remain floating point.

## Example

For the hostile sentence that exposed the failure, the precision-first output
should preserve the transcript without inventing auxiliary evidence:

```text
[affect uncertain] I hate you, I hate you so much—never call me again.
```

The model itself emits validated schema-v2 JSON. Affect is a probability
distribution over perceived vocal expression, never a claim about the
speaker's true internal state.

## Current scope

- English, single-speaker clips or user turns.
- 16 kHz mono PCM16 WAV input, 0.5–30 seconds.
- CTC transcript and CTC-derived word times.
- Localized `laugh`, `cough`, and `throat_clear` events at confidence >=0.98.
- Weak utterance event-presence and style output disabled pending stronger data.
- One calibrated affect distribution per analysed utterance.
- Abstention and OOD probability.
- Valence, arousal, and dominance explicitly unavailable in v0.1.

Cadence v0.1 does not localize changing emotions inside a continuous
utterance. Separate VAD/turn segments can receive separate affect results; an
unsegmented delivery change may produce a mixed distribution or abstention.

## Technical results

| Metric | Full precision | Mixed INT8 |
|---|---:|---:|
| WER | 0.0734 | 0.0782 |
| Localized-event segment macro-F1 at >=0.98 | 0.6645 | 0.6548 |
| Speech-control auxiliary false-positive rate (189 clips) | 0.0000 | 0.0000 |
| Speech-control localized false events/minute | 0.0000 | 0.0000 |
| Event-presence macro-F1 | 0.8258 | 0.8220 |
| Style macro-F1 | 1.0000 | 1.0000 |
| Supported-class affect macro-F1 | 0.4746 | 0.4591 |
| OOD F1 | 0.9907 | 0.9747 |
| Acoustic Preference Score | +0.2727 | +0.2727 |

The local Mac CPU benchmark measured mixed-INT8 RTF 0.0534 and p95 latency
415 ms over 40 contract-valid clips. These are small, partly acted/synthetic
technical evaluations—not evidence of broad naturalistic emotion understanding.
Event-presence and style metrics are shown only as research diagnostics. Those
outputs are disabled in the user-facing runtime because isolated-sound scores
did not establish their reliability on speech.

The table contains internal candidate-selection results. On 480 external
RAVDESS clips, ASR remained accurate at 0.0104 WER while affect macro-F1 fell
to 0.1213 and predictions collapsed toward anger. This external failure
supersedes the internal affect gate and blocks the v0.1 model from release.

## Files

- `attune-cadence-241m-int8.onnx` — recommended mixed-INT8 deployment graph.
- `int8-calibration.json` — thresholds, temperatures, and abstention settings.
- `attune-output-v2.0.schema.json` — authoritative output contract.
- `quantization.json` — exact quantization method and preserved FP nodes.
- `release-gates.json` — executable technical gate results.
- `artifact-manifest.json` — release hashes.

The ONNX graph contains the learned model. The current Python inference path
also uses the official SenseVoiceSmall frontend and decoder assets, which must
be obtained separately under the upstream model agreement. See the
[Project Attune repository](https://github.com/buabaj/attune) for installation
and inference commands.

## Important evaluation caveat

The first sealed evaluation exposed excessive ASR drift. The perception
checkpoint and calibration were kept fixed while a frozen two-block ASR tail
was added, then evaluated on the already-opened partition. The corrected ASR
result therefore needs confirmation on a new untouched external set before a
publication-level claim.

The included gate report must state `release_ready: false` until the original
hostile-speech audio passes the manual regression.

## Intended use and safety

Use Cadence as uncertain supplementary evidence for consented speech research,
accessibility, and low-stakes conversational interfaces. Do not use it for
covert monitoring, diagnosis, deception detection, protected-trait inference,
speaker identification, or automated high-stakes decisions. Do not tell users
they “are” an emotion based on this output.

No controlled downstream human study has yet shown that the representation
improves conversational responses.

## Licence and attribution

The model is derived from **SenseVoiceSmall by FunASR/FunAudioLLM**. Its weights
and derivatives are governed by the
[FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE),
which must be reviewed and followed separately from the MIT-licensed Attune
source code. Retain the SenseVoiceSmall model name, source, and author
attribution when using or sharing this derivative.

Training-source terms and the current redistribution review are documented in
the project dataset card and provenance ledger. A private Hugging Face upload
does not by itself represent public-release clearance.
