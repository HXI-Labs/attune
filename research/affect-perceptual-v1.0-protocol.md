# Affect perceptual v1.0 protocol

Cadence v0.9 improved external RAVDESS macro-F1 from 0.3379 to 0.3853 by
sampling only affect-supervised rows, but it still missed the fixed 0.40 floor.
The available CREMA-D subset contained 492 clips, two sentences, and
actor-intended one-hot labels. This experiment replaces that subset with the
official voice-only listener judgements for a larger speaker-disjoint slice.

The CREMA-D `finishedResponses.csv` table contains 73,254 voice-channel ratings
over 7,442 clips. Version 1.0 uses 4,097 valid clips from 50 actors, with 6 to 12
ratings per clip. Actors 1031-1060 and 1071-1080 form the training partition;
actors 1061-1070 retain their existing development role. Actors 1081-1091 from
the opened regression set and actors 1001-1030 from earlier inspections remain
excluded. Two source-documented invalid clips are excluded. The resulting
labels are listener distributions across neutral, joy, distress, anger, fear,
and other; surprise remains zero because CREMA-D did not solicit that response.
The verified source manifest has SHA-256
`91eabf694df67ea7604f9539a37dac12c5ffe0036321380bd4976dc170c181bf`.

The new slice expands the training corpus from two to twelve matched sentences.
It retains neutral lexical controls and same-text paired supervision. CREMA-D
actor intensity is not interpreted as shouting, and the recordings are
explicit whole-clip negatives only for localized vocal events and event
presence. They do not supervise vocal style.

Training starts from v0.9 and updates only `affect_projection` and
`affect_head`. The encoder, frozen ASR route, event head, style branch, pooling,
and auxiliary heads remain unchanged. Corpus-balanced sampling gives the
CREMA-D perceptual slice, SUBESCO, and Thorsten equal total sampling mass. Six
epochs of 8,192 samples are allowed with a `2e-5` learning rate and early
stopping after three stale validation losses. No regression or external score
may select a checkpoint, threshold, or epoch.

The candidate is retained only if development affect macro-F1 is at least
0.64, opened-regression macro-F1 is at least 0.6236, RAVDESS macro-F1 reaches
0.40 without any predicted class exceeding 45% of clips, APS stays positive,
and all non-affect checkpoint tensors remain byte-identical to v0.9. The fresh
British confirmation set remains sealed regardless of this experiment's
result.

The merged 15,389-row training manifest has SHA-256
`329be1a98e632b91823d15eacb8762990dcd29050acd28aa777d9d590f1905fb`.
Its feature audit found 15,389 unique paths, no known-speaker overlap among
partitions, and 12,597 affect-supervised clips. The audit was completed before
training began.
