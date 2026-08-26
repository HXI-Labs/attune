# Ontology

The ontology describes audible expression, not a speaker's true internal state.
Annotators and product copy ask **“How does the speaker sound?”**, never “What
is the speaker truly feeling?”

## Continuous vocal styles

Time spans may overlap and may be revised during streaming:

- `shouting`
- `whispering`
- `crying_speech`
- `laughing_speech`
- `singing`
- `strained_speech`

Use a style only while it modifies spoken vocal production. Non-speech
occurrences belong to discrete events.

## Discrete vocal events

- `laugh`
- `sob`
- `sigh`
- `cough`
- `throat_clear`
- `sneeze`
- `breath`

Events have temporal extents even when treated as point-like in an interface.
`after_word_id` anchors a between-word or trailing event without embedding it in
transcript text.

## Affect

Valence, arousal, and dominance each use `[-1, 1]` with separate confidence.
Categorical output is a full probability distribution over:

`neutral`, `joy`, `distress`, `anger`, `fear`, `surprise`, `other`, `ambiguous`.

These labels summarize perceived vocal affect. They do not establish internal
emotion. `ambiguous` captures conflicting plausible perceptions; `other`
captures audible affect outside the compact set. When evidence is inadequate,
set `abstain=true`, `top_label=null`, and retain the probability distribution.

## Boundaries and status

All timestamps are non-negative milliseconds and end is never before start.
Annotation lifecycle is `provisional`, `revised`, `committed`, or `retracted`.
Overlaps are valid and must not be forced into nested markup.
