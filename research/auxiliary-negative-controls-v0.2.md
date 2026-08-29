# Auxiliary negative-control experiment v0.2

## Decision

Keep the hardened v0.1 runtime policy. Do not deploy the experimental adapter
or re-enable sneeze, cough-presence, shouting, or whispering from this evidence
alone.

The experiment substantially improves offline separation, but it still emits a
false sneeze on one sealed neutral CREMA-D speech clip. Positive style evidence
also remains isolated FSD50K shout/whisper audio rather than speech embedded in
lexical utterances. This is useful progress, not release evidence.

## Motivation

The first candidate masked ordinary speech for event-presence and style losses.
Its auxiliary heads therefore learned positive source classes without learning
that normal lexical speech should usually produce no event/style output. The
live hostile-speech failure exposed this omission while ASR remained accurate.

## Data contract

The unchanged v0.1 feature manifest has SHA-256
`00fded47e4bb177af4816cc07cd8939f8376e6c77c423e17cd7624789c03c708`.
The opt-in control policy generates a separate manifest with SHA-256
`fc3356f5253280b4f1ba65706b3eef6f79584836815f45b22d5eb3c848f162b4`.

Explicit weak negatives added:

| Dataset/task | Rows |
|---|---:|
| Common Voice event presence | 600 |
| Common Voice shouting/whispering | 600 |
| CREMA-D discrete event presence | 492 |

The split remains 1,691 train / 398 development / 401 sealed. Development has
182 speech controls; sealed has 189. CREMA-D style labels remain masked because
acted intensity is not valid shouting ground truth.

## Lightweight correction

A deterministic balanced logistic adapter was fitted on 32-dimensional frozen
OOD/affect embeddings plus the original seven event-presence and two style
logits. Regularization and thresholds were selected on development only.
Deployment allowlists remain empty in the adapter artifact.

| Metric | Development | Sealed |
|---|---:|---:|
| Event-presence macro-F1 | 0.9105 | 0.8152 |
| Event speech-control false-positive clips | 0 / 182 | 1 / 189 |
| Isolated style macro-F1 | 1.0000 | 1.0000 |
| Common Voice style false-positive clips | 0 / 62 | 0 / 57 |

Sealed event F1 by label:

| Label | F1 | TP | FP | FN | Speech-control FP |
|---|---:|---:|---:|---:|---:|
| laugh | 0.9174 | 50 | 3 | 6 | 0 |
| sob | 0.6667 | 4 | 0 | 4 | 0 |
| scream | 0.9333 | 7 | 0 | 1 | 0 |
| sigh | 0.9697 | 16 | 1 | 0 | 0 |
| cough | 0.6222 | 28 | 6 | 28 | 0 |
| throat_clear | 0.7304 | 42 | 16 | 15 | 0 |
| sneeze | 0.8667 | 13 | 1 | 3 | 1 |

The false sneeze is
`crema-joint-1087_ieo_neu_xx`, neutral lexical speech, at probability 0.9976.
That failure is directly relevant to the user-reported false sneeze and blocks
deployment even though aggregate F1 is strong.

## Required next evidence

1. Retest the original user recording through the protected hardened demo.
2. Collect or obtain consented speech-embedded shouting and whispering positive
   examples with speaker-disjoint evaluation.
3. Add a new untouched ordinary/expressive speech-control set. The opened v0.1
   sealed partition cannot be reused for adapter selection.
4. Retrain/evaluate a candidate on the new lineage, then keep only labels whose
   false-positive risk passes before quantization and after INT8 export.

No result in this experiment changes `release_ready: false` or authorizes a
Hugging Face upload.
