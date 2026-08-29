# Auxiliary negative-control policy

## Why this exists

The first Cadence v0.1 candidate trained event-presence and style heads only on
positive or counter-class isolated sounds. Ordinary Common Voice and CREMA-D
speech was masked for those tasks. Consequently, the heads learned to separate
isolated source classes but were never penalized for emitting cough, sneeze, or
whispering on normal lexical speech.

The live hostile-speech failure made this omission visible. The transcript was
accurate while the auxiliary heads emitted unsupported tags.

## v0.2 opt-in policy

`--auxiliary-negative-controls` on `build_joint_source_manifest.py` applies the
versioned policy in
`configs/data/auxiliary-negative-controls-v0.2.json`:

- Common Voice read speech receives explicit all-negative targets for discrete
  event presence and the current shouting/whispering style inventory.
- CREMA-D speech receives explicit all-negative targets for discrete event
  presence only.
- CREMA-D style targets remain masked. Its acted intensity is not shouting or
  whispering ground truth.
- Existing DCASE, FSD50K, and VocalSound annotations are preserved.

Each affected row carries `auxiliary_negative_tasks`, making the assumption
machine-readable. Empty target lists are therefore distinguishable from missing
annotations.

## Claim boundary

These are weak negative controls, not exhaustive hand annotation. A rare
incidental event may be mislabeled. They are appropriate for penalizing the
catastrophic ordinary-speech shortcut, but they do not establish positive
speech-embedded shouting or whispering performance. Those styles remain
disabled in the release runtime until positive natural speech-style recordings
are collected and evaluated speaker-disjointly.

The original v0.1 manifests and their hashes are unchanged. Applying this
policy creates a new experiment lineage and must never be presented as the
original sealed evaluation.
