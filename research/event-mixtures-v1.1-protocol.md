# Inline-event mixture v1.1 protocol

The v0.8 temporal event head eliminated false positives on opened speech controls,
but its external WESR recall remained 22.9%. The training data contained synthetic
office scenes and isolated utterance-level event labels, not speech turns with
strongly located vocal events.

Version 1.1 adds deterministic mixtures of reviewed Common Voice speech and the five
supported VocalSound classes: laugh, sigh, cough, throat clear, and sneeze. Each
Common Voice clip is paired once with each class. Events are placed before speech,
overlaid within speech, or appended after speech. Event level and final waveform peak
vary independently of event class to reduce reliance on raw amplitude. Only train and development
source partitions are used; the existing sealed rows remain untouched.

The generated source manifest has SHA-256
`edaa669a9b18d77490afa37ef3d21913fecb7b2c3eed51f1ae839227b935040b`.
It contains 2,715 mixtures: 2,405 train and 310 development examples, with
exactly 481/62 examples per event class. Placement is also balanced: 931
before, 906 overlay, and 878 after. The five event levels contain between 534
and 547 examples each, independently of class. Ten silent VocalSound files were rejected
and recorded in the generation audit. Exact source pairings and synthesis
parameters are deterministic from the pinned manifests and generation script.

The spans are exact synthesis envelopes after deterministic active-region trimming.
They are stronger supervision than whole-clip labels but are not human-reviewed event
boundaries or natural-conversation evidence. The generated audio inherits
VocalSound's CC-BY-SA-4.0 terms and remains outside git.

Training will begin from the best accepted affect checkpoint and update only the
857,877-parameter temporal event head. Acceptance requires:

- WESR temporal presence macro-F1 of at least 0.34, up from 0.2882.
- WESR temporal recall of at least 0.30, up from 0.2288.
- WESR false-positive rate no higher than 0.06.
- Opened regression segment macro-F1 of at least 0.60.
- No more than two localized false-positive clips across the 549 opened speech controls.
- All non-event-head tensors byte-identical to the initial checkpoint.

Synthetic development results are diagnostic only. The candidate is rejected if it
passes synthetic validation while missing these opened external gates.

The prepared 18,104-row feature manifest has SHA-256
`844eccce5ede16c206df96a5ad32f5c853dc9f17de4949ad9d1b1cb3a1c70e98`.
It contains 7,997 train and 1,575 development rows with explicit strong-event
targets, including empty speech controls. The integrity audit found 18,104
unique feature paths and no known-speaker overlap among partitions.

The run permits eight epochs of 8,192 corpus-balanced samples, with physical
batches of six, four-step gradient accumulation, a `5e-5` head learning rate,
and early stopping after three stale development losses. No encoder or
non-event head is trainable.
