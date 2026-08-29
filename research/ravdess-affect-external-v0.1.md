# RAVDESS external affect audit v0.1

## Decision

The current Attune Cadence v0.1 affect branch is not release-ready. Preserve
the accurate ASR path, keep the existing runtime abstention/suppression policy,
and evaluate the predeclared affect-only adapter in
`configs/model/affect-adapter-v0.3.json` before considering any affect output
for deployment.

## Evaluation boundary

This is an external evaluation-only slice of RAVDESS v1.0.0:

- 480 clips from actors 17–24 (60 clips per actor);
- two fixed, lexically neutral statements;
- seven mapped Attune categories across normal/strong acted delivery;
- exact official archive MD5 verification;
- explicit 16 kHz mono PCM16 normalization; and
- no use for fitting, calibration, threshold selection, or release-weight
  training.

RAVDESS is used under CC BY-NC-SA 4.0 for non-commercial research inspection.
Its acted source labels are perceived-expression evaluation targets, not proof
of any speaker's internal state.

Reproducibility hashes:

| Artefact | SHA-256 |
|---|---|
| Official speech archive | `5d208e01632cc3e5242106fa2af3273e6dc5239fb8143131979ac74c4aa40657` |
| Committed evaluation manifest | `1312eb20eca0c1ad2d5fbc4d6c56c98a263b01871f8fb68960b6128fa2acdd1b` |
| INT8 model | `d9b36f099aa456c6f1a1348099cb375d36b34433c898ef31fdbf7a24fe27e9aa` |
| Frozen calibration | `3777c2029d4e994716458de250b431772de3484d22603d4c76ef245b7a00e83e` |
| Raw external scores | `a339512fb154a575652e4108887f3a8b07c3c91f33cca7f0238a9770823e9135` |

## Baseline result

| Metric | Result |
|---|---:|
| ASR WER | 0.0104 |
| Affect macro-F1 (present classes) | 0.1213 |
| Affect Brier score | 1.0286 |
| Affect ECE | 0.4107 |
| Affect coverage | 0.9604 |
| Full-coverage error | 0.8000 |
| Selective error | 0.8048 |
| Acoustic Preference Score | +0.1536 |
| Localized-event false-positive clips | 2 / 480 |
| Event-presence/style false-positive clips | 0 / 480 |

ASR is strong. Affect is not. Selective prediction does not reduce error, so
the current confidence threshold is not a meaningful safety control on this
cross-corpus set.

## Class-collapse diagnosis

| Target | Correct / support | Recall |
|---|---:|---:|
| anger | 61 / 64 | 0.9531 |
| neutral | 30 / 96 | 0.3125 |
| other | 3 / 64 | 0.0469 |
| fear | 1 / 64 | 0.0156 |
| joy | 1 / 64 | 0.0156 |
| distress | 0 / 64 | 0.0000 |
| surprise | 0 / 64 | 0.0000 |

The model predicts anger for 340 of 480 clips (70.8%), including 55/64 joy,
59/64 fear, 60/64 disgust/other, and 52/64 surprise clips. This is a class
collapse, not an ontology-edge disagreement. Temperature scaling cannot repair
the ranking of class logits.

## Predeclared next candidate

The next and only candidate in this lineage is a balanced multinomial logistic
adapter over the frozen 32-dimensional acoustic/OOD embedding, seven affect
logits, and three VAD predictions. Regularization, temperature, and abstention
are selected only on the original development partition. CTC logits, encoder
weights, transcript decoding, event heads, and style heads remain byte-for-byte
unchanged.

The adapter remains disabled unless it reaches at least 0.40 macro-F1 on this
external set, retains at least 0.70 anger recall, avoids any predicted class
exceeding 60% of clips, covers at least 50%, and lowers selective error. Passing
these gates would justify further evaluation, not release by itself.
