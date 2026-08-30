# ONNX batch-padding audit

Release score collection remains single-item. A real-model parity audit found
that multi-item padding changes temporal event logits near the end of shorter
clips.

## Audit inputs

- Model: `attune-berst-style-v0.1-fp.onnx`, SHA-256
  `3b2eae1ac884f2259b97409f5470b1b0f91494fc70be08c62a5b1224d1cfe875`.
- Manifest: 100-speaker British controls, SHA-256
  `cc1c6521c72e780eefb569fbe8963483b40d66f505537f7eb94f82a65b0c1923`.
- Batch-one score SHA-256:
  `8bdc9b9ed67d75cca9d2377a2db93260141686a42f05434874b66d3607424816`.
- Batch-four score SHA-256:
  `c2ef36d54cbb3b7753c571c0f0fecc87372a2473fe4dbb0788ba7f69893be64d`.

Both passes emitted 100 rows in the same clip order. Maximum absolute output
differences were:

| Output | Maximum difference |
| --- | ---: |
| Event frame logits | 34.3703 |
| Event-presence logits | 2.26e-6 |
| Style logits | 1.91e-6 |
| Affect logits | 2.62e-6 |
| VAD | 1.28e-6 |
| OOD embedding | 5.70e-7 |
| OOD logit | 3.34e-6 |

The largest event difference occurred on the final valid frame of
`cv17-british-29ad5c7dace5addc`. Removing the final one, two, and three frames
reduced the maximum event-logit difference to 12.5245, 4.1081, and 0.000237,
respectively. This pattern is consistent with padded encoded frames entering
the temporal convolution's boundary receptive field. It does not establish
that the encoder itself is the source.

## Decision

The proposed batched collector was not merged. Calibration, candidate
selection, and release evaluation must use batch size one. Batch-throughput
figures may be reported only as performance diagnostics until a padding mask is
applied inside the temporal head and multi-item parity is covered by a real
ONNX regression test. This audit did not change any model-selection result.
