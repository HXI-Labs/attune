# Model card: Attune Cadence 242M

**Release ID:** `attune-cadence-v0.1`

## Status

This is a private research release candidate. It is not cleared for public
weight publication. Public release still requires a fresh consented human
regression recording, affect confirmation, and approval of the derivative
weight redistribution review.

Cadence v0.1 deliberately disables vocal styles. An FSD50K-derived head
hallucinated `whispering` on four ordinary speech controls, and a replacement
BERSt shouting head reached only 0.689 F1 and 0.551 recall at the fixed
false-positive limits. Neither head is part of the release candidate.

## Intended task

For English, single-speaker, 0.5–30 second, 16 kHz mono PCM16 WAV input, emit:

- a CTC transcript and approximate word timings;
- conservative `laugh`, `sigh`, `cough`, `throat_clear`, and `sneeze` events;
- a probability distribution over perceived affect categories;
- calibrated abstention and out-of-distribution information; and
- schema-v2 JSON plus deterministic XML.

The system estimates audible expression. It does not establish a speaker's
internal emotional state.

## Architecture

The ONNX graph is derived from SenseVoice-Small and contains 241,904,650
parameters. One encoder pass produces CTC, localized-event, affect, OOD, and a
padding-safe 5,120-value acoustic embedding. A 30,726-parameter NumPy linear
head consumes the embedding for utterance-level event decisions. The combined
deployment contains 241,935,376 parameters.

The event head has an explicit `none` logit and a validation-selected margin
threshold. Raw SenseVoice AED and emotion tags are not exposed. Existing weak
style and event-presence outputs remain disabled. The small head does not
require PyTorch at inference time.

The full-precision graph is 969,933,834 bytes. ONNX Runtime dynamic per-channel
QInt8 reduces it to 500,481,798 bytes. The event head and frontend assets are
separate files.

## Training and calibration

The retained graph is the affect-focus v0.9 candidate. ASR uses a frozen
base-ASR view so perception adaptation cannot change the CTC path. The event
head is a frozen-encoder linear probe trained from VocalSound, with FSD50K and
CREMA-D controls used for abstention. Its English-query embedding is verified
against the full-precision ONNX output on 20 clips with maximum absolute error
`5.34e-5`.

All datasets and base weights have independent licence and provenance records
under `data/provenance/`. Source labels in the event inspection sets are weak
labels rather than reviewed natural inline-event annotations.

## Evaluation

| Metric | Full precision | INT8 |
|---|---:|---:|
| VocalSound event macro-F1, 80 opened clips | 0.814 | 0.826 |
| External OOD false-positive rate, 160 clips | 0.0125 | 0.0125 |
| Exact hostile-lexical transcripts | 4/4 | 4/4 |
| Events/styles on hostile-lexical controls | 0/0 | 0/0 |
| Affect abstention on hostile-lexical controls | 4/4 | 4/4 |

On the retained full-precision graph, affect macro-F1 is 0.6643 on development,
0.6336 on the opened regression set, and 0.3853 on the 480-clip external
RAVDESS set. INT8 reaches 0.3803 on the same set, an absolute loss of 0.0051;
selective risk still improves and coverage is 0.75. Quantization is within the
two-point degradation budget, but both graphs remain below the project's 0.40
public-release target.

The real FastAPI and pseudo-streaming smoke test passed on the development Mac.
For a 3.74-second regression clip it measured 207 ms processing time and RTF
0.055. This is a single smoke measurement, not a hardware benchmark.

The four hostile-lexical controls use neutral system voices. They prove that
the exact words no longer trigger the original false tags, but they do not
replace the required consented human recording or expressive-speech testing.

## Known limitations

- Vocal styles are disabled.
- Event results are based partly on isolated, source-labelled vocal sounds and
  do not establish natural inline-event localization quality.
- Affect supervision remains dominated by acted and categorical data.
- Affect is utterance-level; changing emotion inside one uninterrupted turn is
  not localized.
- Valence, arousal, and dominance are unavailable in v0.1.
- Word timestamps are approximate CTC alignments.
- Ghanaian, British, and other accent slices are too small for a fairness claim.
- Noise, far-field speech, overlap, sarcasm, code-switching, atypical voices,
  and unseen devices can produce confident errors.
- No downstream human interaction study has established conversational value.

## Intended and prohibited use

Cadence is intended for consented speech research, accessibility experiments,
and low-stakes conversational interfaces where uncertainty is preserved. It
must not be used for covert monitoring, diagnosis, deception detection,
speaker identification, protected-trait inference, or automated high-stakes
decisions. Applications must not present perceived affect as a verified fact.

## Third-party licences

Project code is MIT licensed. SenseVoice-Small weights and their derivatives
are governed separately by the
[FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).
Dataset terms are recorded under `data/provenance/`. The committed
redistribution review currently does not approve public weight publication.
