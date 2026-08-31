# Attune release-candidate dataset card

## Status and claim boundary

This project uses several head-specific multi-corpus bundles, not one new gold
dataset. Every label retains its source status. Audio is not committed or
redistributed by this repository. Feature/audio hashes and pinned source
revisions are recorded in manifests and `data/provenance/`.

## Original prototype bundle

| Source | Purpose | Split rule | Important limitation |
|---|---|---|---|
| Common Voice 17 English (600) | CTC replay | one clip/client; deterministic 80/10/10 train/development/sealed; British inspection clients excluded | transcript only |
| DCASE 2016 Task 2 (316) | laugh/cough/throat-clear timing and presence | source-recording-disjoint train/development; public test sealed | synthetic scenes |
| FSD50K bounded slice (320) | shout/whisper style and sob/scream presence | source train plus label-stratified development/sealed; prior inspection excluded | weak clip labels; no timing claim |
| VocalSound (762) | laugh/sigh/cough/throat-clear/sneeze presence | speaker-disjoint train/development; 80 prior inspection clips sealed | acted weak clip labels; no timing claim |
| CREMA-D paired v0.1 (492) | categorical affect and same-text/different-delivery | actors 1031–1050 train, 1061–1070 development, 1081–1091 sealed | acted one-hot source labels |
| MSP-Podcast v2 | future naturalistic affect and V/A/D | unavailable pending owner access terms | absent from current bundle |

`surprise` and `ambiguous` remain schema categories but have no positive
CREMA-D class. V/A/D must be marked unavailable in a release trained without
MSP-Podcast or another reviewed dimensional source.

The complete feature manifest has 2,490 rows (1,691 train, 398 development,
401 sealed test) and SHA-256
`00fded47e4bb177af4816cc07cd8939f8376e6c77c423e17cd7624789c03c708`.

The corresponding source-audio manifest SHA-256 is
`685d3353f79a3567c9e64a58b93854cdb6f44bbfac09bcbbeae56cc5d6e527de`.
The release artifact manifest records both files and the audit report.

## Subsequent candidate sources

Later isolated event, style, and affect branches add the following sources.
They do not silently change the original prototype split or convert weak
labels into gold annotations.

| Source | Prepared clips | Purpose | Split rule | Important limitation |
|---|---:|---|---|---|
| SUBESCO v1.1 | 7,000 | auxiliary acoustic affect | 14/2/4 actors across train/development/sealed | Bangla acted intended labels; no English ASR supervision |
| Thorsten Emotional v2 | 2,399 | auxiliary affect and whisper experiments | 210/45/45 sentences across train/development/sealed | one German speaker; cannot establish English or speaker generalization |
| BERSt | 4,441 | English shouting, affect, ASR replay, device/gain robustness | official 78/10/10-speaker train/development/sealed split | actor prompts are intended delivery, not listener distributions |
| DisfluencySpeech | 5,000 | weak event-presence training | official 4,500/250/250 sentence-disjoint clips | one English speaker; no timestamps inferred from word-position tags |
| Attune inline-event mixtures | 2,715 | synthetic strong event-span augmentation | Common Voice speaker-disjoint train/development sources; sealed sources excluded | exact synthesis envelopes are not human event boundaries |

SUBESCO and BERSt are CC BY 4.0; Thorsten Emotional is CC0 1.0;
DisfluencySpeech is Apache-2.0; inline-event mixtures combine CC0 Common Voice
speech with CC BY-SA 4.0 VocalSound events and retain the share-alike condition.
Exact source hashes, attribution, and redistribution notes are in the
corresponding provenance records.

## Supervision semantics

- DCASE onset/offset annotations may supervise localized spans.
- FSD50K isolated-event labels supervise only utterance presence.
- FSD50K shout/whisper labels supervise only utterance-scope styles.
- CREMA-D categories are acted source labels, not Attune consensus or verified
  internal emotion.
- SUBESCO and BERSt affect categories are acted or prompted intended-delivery
  labels, not listener-verified perceived affect.
- BERSt `shout` supervises only `shouting`; `no-shout` is an explicit negative.
- DisfluencySpeech transcript tags supervise utterance event presence only.
- Inline-event mixtures supervise their deterministic insertion spans but are
  never external evaluation evidence.
- DCASE/FSD50K are known OOD examples for affect, but remain in-domain for
  their event/style tasks.
- Missing task annotations are masked, never converted to negatives.

An opt-in v0.2 experiment adds explicitly marked weak speech-negative controls;
it does not alter this v0.1 manifest. See
`docs/auxiliary-negative-controls.md` and
`configs/data/auxiliary-negative-controls-v0.2.json`.

## Leakage controls

The source validator rejects duplicate IDs and known speakers crossing splits.
DCASE groups source recordings. Common Voice uses unique clients. CREMA-D,
SUBESCO, and BERSt use disjoint actors. BERSt preparation also excludes all 82
rows in 31 repeated-waveform groups, including a group that crossed official
development and test partitions. FSD50K lacks speaker identity. Thorsten and
DisfluencySpeech contain one speaker each, so only sentence/clip disjointness
can be claimed for those corpora.

## Rights, privacy, and release

Each `data/provenance/*.yaml` is authoritative for licence and consent-review
status. CREMA-D remains local under reviewed ODbL/DbCL conditions. SenseVoice
weights and trained deltas are governed separately. No audio or weights become
public merely because project code is MIT.

## Final-evaluation use

The 401-row sealed partition was first opened only after perception checkpoint,
calibration, and mixed-precision selection. That pass exposed excessive ASR
drift in the shared adapted encoder. A frozen two-block ASR tail was then added
without changing perception logits or thresholds, validated on development,
and evaluated on the already-opened partition. The final ASR score therefore
needs confirmation on a new untouched external set. The perception branch was
not retuned from sealed labels.

The current sources do not establish naturalistic cross-corpus affect,
multi-rater ambiguity, subgroup fairness, or broad accent robustness. Most
affect labels are acted, prompted, and one-hot. DCASE timing and inline-event
mixtures are synthetic; FSD50K, VocalSound, and DisfluencySpeech labels are
weak or acted. These limitations are part of the evidence, not future labels
to infer away.
