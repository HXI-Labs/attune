# CC0 acoustic adapters v0.4

## Decision

Reject both candidates and keep deployment disabled. The additional CC0
same-text expressive speech improves cross-corpus affect classification and
adds genuine lexical whisper examples, but it does not clear the predeclared
release gates.

The accurate CTC path was not trained or exported during this experiment. All
2,399 Thorsten rows retained their German transcript for provenance while
setting `token_ids` to null.

## Data contract

Thorsten-Voice Emotional v2.0 is CC0 and contains one German male speaker
performing the same 300 sentences in eight deliveries. The archive contains
2,399 WAV files rather than the documented 2,400: whisper recording
`2cc2cc4a34b961ef1657cc82dbd18875.wav` is absent.

The split is sentence-disjoint:

| Partition | Sentences | Clips |
|---|---:|---:|
| Train | 210 | 1,679 |
| Development | 45 | 360 |
| Sealed test | 45 | 360 |

Angry is mapped only to affect `anger`; it is never treated as `shouting`.
Whisper is mapped to the `whispering` style. Sleepy and acted-drunk deliveries
remain masked for affect.

## Affect adapter

The v0.4 affect adapter combines the original training lineage with Thorsten
training rows. Regularization, temperature, and abstention are selected on the
combined original and Thorsten development rows. RAVDESS was already opened by
v0.3, is not used for parameter selection, and is reported only as a repeated
external diagnostic.

| Partition | Macro-F1 | Accuracy | Coverage | Full error | Selective error |
|---|---:|---:|---:|---:|---:|
| Original development | 0.5122 | 0.5167 | 0.6583 | 0.4833 | 0.4177 |
| Original opened sealed | 0.5071 | 0.5303 | 0.7121 | 0.4697 | 0.3830 |
| Thorsten development | 0.5633 | 0.5600 | 0.4222 | 0.4400 | 0.3053 |
| Thorsten sealed | 0.5017 | 0.5067 | 0.5022 | 0.4933 | 0.3628 |
| External RAVDESS | 0.2792 | 0.3625 | 0.7021 | 0.6375 | 0.5697 |

External RAVDESS macro-F1 improved from v0.3's 0.1588 to 0.2792, but remains
below the fixed 0.40 gate. Original development also fell below its 0.55
retention gate. The candidate is rejected.

## Whisper adapter

The whisper adapter is a balanced binary logistic model over the frozen
32-dimensional acoustic/OOD embedding and the two original style logits.

| Partition | Recall | Precision | F1 | Negative FPR |
|---|---:|---:|---:|---:|
| Original development | 0.8750 | 0.8750 | 0.8750 | 0.0143 |
| Original opened sealed | 0.6250 | 1.0000 | 0.7692 | 0.0000 |
| Thorsten development | 0.8444 | 0.9048 | 0.8736 | 0.0889 |
| Thorsten sealed | 0.9556 | 0.7818 | 0.8600 | 0.2667 |
| External British ordinary speech | n/a | n/a | n/a | 0.0000 |

The adapter preserves zero false positives on the 100-speaker British negative
control, but English held-out whisper recall is only 0.625 and Thorsten sealed
neutral FPR is 0.2667. It is rejected and remains disabled.

## Interpretation and next step

The linear corrections are capacity- and domain-limited. The direction is
positive, but one German speaker cannot establish robust affect or whisper
generalization. The next data stage should add a permissively licensed,
speaker-diverse emotional corpus while keeping ASR frozen. SUBESCO is the
current candidate: its official Zenodo record declares CC BY 4.0, 7,000 Bangla
utterances, 20 actors, ten sentences, and seven emotion categories.

No v0.4 result changes `release_ready: false`.

Artefact hashes:

- affect adapter: `eaf4b5d8ee4568d361bad1faff70726c4515485d00e6eaed99539fb1121e6133`
- affect report: `823f7eefaca6b6dc4e0fcbec92d2f252a06531c716186f8ff2f7efa3bc4bc8cd`
- whisper adapter: `19bebd5ec088ea7c03546eb3c34a6f597386acba138ec29ed3d920098778decc`
- whisper report: `05ebc70e4c4b2efc2048ee467df927c7cbbe64d9ff7068c667295544225ed3f1`
