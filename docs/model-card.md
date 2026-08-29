# Model card: Attune Cadence 241M

**Release ID:** `attune-cadence-241m`

**Tagline:** Hear how it was said.

## Status

This is an accuracy-hardening prerelease, not a public-weight release. A live
test found severe false-positive style and event tags while ASR remained
accurate. The demo and publication were withdrawn. The executable report in
`artifacts/release/v0.1/release-gates.json` is deliberately
`release_ready: false` until the original hostile-speech case is retested.
Public redistribution also depends on a final review of every training
source's derivative-artifact terms.

The candidate has 241,609,098 parameters. Its trained delta is
`artifacts/training/local-upper-two-v0.1/model.pt`; its deployment graphs are:

- FP: `artifacts/models/attune-split-tail-v0.1-fp.onnx` (968,740,218 bytes)
- mixed INT8: `artifacts/models/attune-split-tail-v0.1-int8.onnx`
  (594,795,068 bytes)

All release hashes are recorded in
`artifacts/release/v0.1/artifact-manifest.json`.

## Intended task

For English, single-speaker, 0.5–30 second, 16 kHz mono PCM16 WAV input, emit:

- CTC transcript;
- localized `laugh`, `cough`, and `throat_clear` spans at confidence >= 0.98;
- no user-facing weak event-presence or style labels until they are validated
  against speech-negative controls;
- a calibrated distribution over perceived affect categories;
- abstention and out-of-distribution information; and
- schema-v2 JSON plus deterministic XML.

The system estimates audible expression, not verified internal emotion.

## Architecture

The lower SenseVoice encoder is shared. The selected upper-two adapted path
feeds all paralinguistic heads. CTC ASR uses frozen copies of only the two
original upper encoder blocks and final norms, followed by the shared frozen
temporal-predictor blocks. This 6,318,080-parameter copy corrected ASR drift
without changing any selected perception logits; the full model stays below
300M parameters.

The deployment graph uses ONNX Runtime dynamic per-channel QInt8 for most
eligible shared weights. The upper eight perception blocks, two-block frozen
ASR tail, pooling, and small utterance heads remain FP to meet fixed parity
gates. “INT8” is therefore a deployment shorthand for a mixed-precision graph,
not a claim that every operator is integer-only.

## Training data and procedure

The source-labelled bundle contains 2,490 rows / 3.79 hours: 1,691 train, 398
development, and 401 sealed test. It combines Common Voice CTC replay, DCASE
strong event timing, weak FSD50K/VocalSound event/style labels, and acted
CREMA-D categorical affect pairs. Known speakers and recording groups are
disjoint where source metadata permits. Missing labels are masked.

The frozen-head model was trained first. The selected upper-two run warm-started
only its task heads, trained for 14 epochs, early-stopped after three stale
epochs, and selected epoch 11 at development loss 1.2537. It updates 7,609,931
parameters. Training ran locally on CPU at zero cloud cost; no rented GPU was
used.

No reviewed V/A/D source was available. The architecture retains a dormant
head, but runtime calibration marks dimensional outputs unavailable.

## Evaluation

### Sealed technical results

| Metric | FP | mixed INT8 | Fixed requirement |
|---|---:|---:|---:|
| WER | 0.0734 | 0.0782 | FP ≤ base + 0.01; INT8 ≤ FP + 0.005 |
| Localized-event segment macro-F1 at >= 0.98 | 0.6645 | 0.6548 | FP ≥ 0.50; loss ≤ 0.02 |
| Speech-control auxiliary false-positive rate | 0.0000 | 0.0000 | ≤ 0.01 |
| Speech-control localized false events/minute | 0.0000 | 0.0000 | ≤ 0.10 |
| Event-presence macro-F1 | 0.8258 | 0.8220 | FP ≥ 0.50; loss ≤ 0.02 |
| Style macro-F1 | 1.0000 | 1.0000 | FP ≥ 0.60; loss ≤ 0.02 |
| Supported-class affect macro-F1 | 0.4746 | 0.4591 | FP ≥ 0.40; loss ≤ 0.02 |
| OOD F1 | 0.9907 | 0.9747 | FP ≥ 0.75; loss ≤ 0.02 |
| Affect coverage | 0.8258 | 0.8712 | FP ≥ 0.50 |
| Acoustic Preference Score | +0.2727 | +0.2727 | > 0 |

FP affect error falls from 0.4773 at full coverage to 0.4128 under the
development-selected abstention rule. FP Brier score is 0.5999 and ECE is
0.1134. The categorical metric covers the six supported classes (`neutral`,
`joy`, `distress`, `anger`, `fear`, `other`); ontology-wide F1 including
unsupported classes is lower.

The event-presence and style metrics are offline research diagnostics only.
Those heads are disabled in runtime output because their source data did not
establish acceptable false-positive behaviour on ordinary speech. Six real
speech clips that previously triggered multiple false tags emitted zero events
and zero styles under the hardened policy, with their transcripts preserved.

### Runtime

On the development Mac CPU, mixed INT8 reached 0.0534 RTF over 40 contract-valid
clips, with p50/p95/p99 latency of 206/415/440 ms, 100% JSON/XML validity, and
zero committed retractions. One hundred DCASE source rows at 44.1 kHz were
explicitly excluded from this runtime benchmark because the API contract is
16 kHz mono; their model scores remain present in offline evaluation.

Archived unmodified SenseVoice evaluations report WER 0.2365 on a 100-clip
Ghanaian-English broadcast slice and 0.1149 on a 100-speaker British Common
Voice slice. The final FP ASR route copies those exact frozen base weights, but
the final graph was not rerun because those audio caches were not retained; the
mixed-INT8 route has not been measured on either slice. These archived results
are context, not final-model accent validation.

## Evaluation caveat

The first sealed pass used the selected single-tail adapted encoder and revealed
2.23 absolute WER points of drift over the frozen base. All other gates passed.
The perception checkpoint, thresholds, and model choice remained fixed; only a
frozen two-block ASR route was added. Development confirmed exact base WER and
unchanged perception metrics, after which the corrected graph was evaluated on
the already-opened sealed set.

This makes the correction transparent and mechanically grounded, but the
corrected ASR result is no longer a pristine one-shot sealed estimate. Confirm
it on a new external untouched English/accent set before publication. The
sealed perception evaluation was not used to retune perception heads.

## Known limitations

- Affect supervision is acted, one-hot CREMA-D—not naturalistic multi-rater
  soft labels.
- Only three event classes are localized; all weak utterance-presence event
  outputs are currently disabled.
- Style output is currently disabled. Its offline F1 is based on a narrow
  weak-label set and cannot be read as natural speech-style reliability.
- Word timestamps group genuine CTC token spans using SentencePiece boundaries;
  they are approximate acoustic alignments, not interpolated word durations.
- V/A/D, `surprise`, and `ambiguous` lack positive reviewed supervision.
- The semantic-conflict set is small; positive APS does not prove lexical
  disentanglement across domains.
- Ghanaian/British manifests and archived base-ASR results exist, but the final
  mixed graph lacks a rerun and the slices are not sufficient fairness or
  cross-accent validation.
- No human interaction study, naturalistic cross-corpus affect test, GPU
  benchmark, Apple Neural Engine benchmark, or production monitoring study has
  been completed.
- Sarcasm, overlap, far-field speech, clipping, noise, code-switching, atypical
  voices, and unseen devices can yield confident errors.

## Prohibited uses

Do not use this model for diagnosis, deception detection, covert monitoring,
protected-trait inference, speaker identification, or automated high-stakes
decisions. Do not tell a user they “are” an emotion based on this output.

## Third-party licences

SenseVoice-Small source is MIT, while official weights use the
[FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).
Use and release must attribute SenseVoiceSmall by FunASR/FunAudioLLM and comply
with the exact agreement. Dataset and baseline-model records are in
`data/provenance/`; project code licensing does not override them.
