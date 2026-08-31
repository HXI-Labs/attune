# Inline-event mixture v1.1 results

Event mixture v1.1 tested whether deterministic speech-plus-vocal-event mixtures
could improve the retained v0.8 event head without reintroducing fabricated event
tags on ordinary speech. The run started from Cadence v0.9 and trained only the
857,877 parameters in `event_head` for eight epochs on Apple MPS.

The candidate is rejected. It preserved the desired conservative behaviour but
did not generalise well enough to natural external event data.

| Gate | Result | Required | Decision |
|---|---:|---:|---|
| WESR temporal-presence macro-F1 | 0.1347 | >= 0.34 | Fail |
| WESR temporal-presence recall | 0.1078 | >= 0.30 | Fail |
| WESR temporal-presence false-positive rate | 0.0000 | <= 0.06 | Pass |
| Opened regression segment macro-F1 | 0.2676 | >= 0.60 | Fail |
| Opened speech-control false-positive clips | 0 / 549 | <= 2 | Pass |
| Checkpoint scope | Event head only | Event head only | Pass |

The synthetic development set produced temporal-presence macro-F1 of 0.3412
with recall of 0.2505 and zero localized false positives across 1,241 speech
controls. The opened regression set produced presence macro-F1 of 0.3051,
recall of 0.3833, and no localized false positives across 549 controls. These
results confirm that mixture training retained strong negative control, but the
large drop on WESR shows that the synthetic positives did not transfer.

The checkpoint scope audit found exactly ten changed tensors, all beneath
`event_head`. The shared encoder, CTC path, style head, and affect head remained
unchanged. The best validation loss was 0.05858 at epoch 8; elapsed training time
was 6,031 seconds.

The retained Cadence v0.9 checkpoint remains the base for subsequent work. Event
v1.1 must not be composed into a release model. Future event work needs natural
speech turns containing time-aligned vocal events rather than more synthetic
mixture volume.

## Reproducibility

- Protocol: `research/event-mixtures-v1.1-protocol.md`
- Training manifest SHA-256: `844eccce5ede16c206df96a5ad32f5c853dc9f17de4949ad9d1b1cb3a1c70e98`
- Initial checkpoint SHA-256: `3e47e9b7974718f2589e0623d40e61e4c7eb6116d614e234694b52866f686e08`
- Rejected checkpoint SHA-256: `fa4f50c689751c32bb135c9dd7a5350bdb825424d350c9ea70cf5f32f3e686d0`
- ONNX export SHA-256: `3aa6eed77ec2cddb60d72209e024bf0c6e2e9d87c66b9dca7d68d9d08095b374`
- Acceptance artifact SHA-256: `6a69c44d8021b8170db85fdd30cfb582c5b51f91eb85ab01288be123187cb126`
- Pre-run code commit: `1542a3a3cb34e6bd89c8a6a11ee70e9230910bb6`

The authoritative reports are stored under
`artifacts/evaluation/event-mixtures-v1.1/` and the checkpoint under
`artifacts/training/local-event-mixtures-v1.1/`.
