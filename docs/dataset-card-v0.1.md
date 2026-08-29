# Attune v0.1 source-labelled dataset card

## Status and claim boundary

This is a multi-corpus bundle, not a new gold dataset. Every label retains its
source status. Audio is not committed or redistributed. Feature/audio hashes
and source revisions are recorded in manifests and `data/provenance/`.

## Components

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

## Supervision semantics

- DCASE onset/offset annotations may supervise localized spans.
- FSD50K isolated-event labels supervise only utterance presence.
- FSD50K shout/whisper labels supervise only utterance-scope styles.
- CREMA-D categories are acted source labels, not Attune consensus or verified
  internal emotion.
- DCASE/FSD50K are known OOD examples for affect, but remain in-domain for
  their event/style tasks.
- Missing task annotations are masked, never converted to negatives.

## Leakage controls

The source validator rejects duplicate IDs and known speakers crossing splits.
DCASE groups source recordings. Common Voice uses unique clients. CREMA-D uses
disjoint actors. FSD50K lacks speaker identity, so its clip/inspection exclusion
is retained and reported as a limitation.

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

The current bundle does not establish naturalistic cross-corpus affect,
multi-rater ambiguity, subgroup fairness, or broad accent robustness. CREMA-D
affect is acted and one-hot; DCASE timing is synthetic; FSD50K and VocalSound
labels are weak or acted. These limitations are part of the evidence, not
future labels to infer away.
