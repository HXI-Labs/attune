# Annotation guide

## Framing

Ask: **“How does the speaker sound in this recording?”** Do not ask what the
speaker truly feels, why they sound that way, whether they are truthful, or
whether they have a health condition. Label only audible evidence and use
uncertainty rather than forced certainty.

## Separate passes

1. Transcribe words and mark word boundaries without seeing affect choices.
2. Mark non-speech vocal events and their temporal extents.
3. Mark continuous vocal-style spans; overlaps are allowed.
4. Rate valence, arousal, and dominance for the stated scope.
5. Assign a **soft** probability distribution across affect categories, then
   choose a top label only when the evidence supports one.
6. Record quality problems, ambiguity, and abstention independently.

Separating passes reduces anchoring between transcript content and vocal
perception. Annotators should replay audio, but written words alone are
insufficient for vocal labels.

## Label rules

- Use `crying_speech`/`laughing_speech` when the behaviour modifies speech and
  `sob`/`laugh` for discrete non-speech events.
- Prefer the shortest defensible style boundary; do not infer beyond audio.
- Preserve disagreements as soft labels. Never manufacture consensus by
  discarding minority ratings.
- Use `ambiguous` for competing perceptions and abstain when audio or ontology
  support is inadequate. Abstention requires `top_label=null`.
- Record clipping, low SNR, far-field speech, language mismatch, and unfamiliar
  expression as possible sources of uncertainty.

## Quality and splits

Use multiple independent annotators, blind adjudication where possible, and
report agreement per label and subgroup. Split datasets by speaker before
examples are exposed to training: train, validation, and test must be
speaker-disjoint. Where source identity permits, also guard against episode,
recording-session, and near-duplicate leakage.

Annotations must retain dataset item IDs, annotator protocol version, ontology
version, timestamps, and provenance. Do not store unnecessary identity or
protected-trait attributes.
