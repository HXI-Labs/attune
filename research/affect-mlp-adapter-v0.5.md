# Speaker-diverse affect MLP v0.5

## Decision

Reject and keep disabled. The nonlinear adapter materially improves external
generalization and retains the original English domain, but it does not clear
the fixed external or new speaker-disjoint SUBESCO gates.

## Protocol

The configuration was committed at `7c90324` before inspecting any SUBESCO
development or sealed metrics. A two-layer 128/64 GELU MLP consumes the frozen
32-dimensional acoustic/OOD embedding, eight affect logits, and three VAD
outputs. Training combines the original lineage, CC0 Thorsten, and CC BY 4.0
SUBESCO with inverse dataset-and-class-group weighting. Early stopping uses the
mean per-dataset development macro-F1.

No encoder, CTC, event, style, ONNX, or INT8 parameter changed.

## Results

| Partition | Macro-F1 | Accuracy | Coverage | Full error | Selective error |
|---|---:|---:|---:|---:|---:|
| Original development | 0.6193 | 0.6333 | 0.8167 | 0.3667 | 0.3367 |
| Original opened sealed | 0.5728 | 0.5985 | 0.8864 | 0.4015 | 0.3504 |
| Thorsten development | 0.5401 | 0.5467 | 0.5689 | 0.4533 | 0.3438 |
| Thorsten opened sealed | 0.4856 | 0.4889 | 0.5289 | 0.5111 | 0.4286 |
| SUBESCO development | 0.3911 | 0.4186 | 0.4357 | 0.5814 | 0.4852 |
| SUBESCO sealed actors | 0.3951 | 0.4179 | 0.2686 | 0.5821 | 0.4043 |
| External RAVDESS | 0.3537 | 0.4396 | 0.6667 | 0.5604 | 0.4781 |

The best epoch was 123, temperature 1.1295, and abstention threshold 0.4246.
RAVDESS macro-F1 improves from 0.1588 in v0.3 and 0.2792 in v0.4 to 0.3537,
while the largest predicted class falls to 0.248. The 0.40 external gate still
fails. SUBESCO sealed macro-F1 is 0.3951 against the fixed 0.60 gate.

## Interpretation

More speakers and a nonlinear correction help, but the frozen 43-feature
bottleneck is insufficient. The next candidate should retrain the attentive
pooling, affect projection, and affect head from the full 512-dimensional
encoder frames while the SenseVoice encoder and CTC path remain frozen. That
step benefits from a rented CUDA GPU but does not require encoder fine-tuning.

No v0.5 result changes `release_ready: false`.

Artefact hashes:

- adapter: `5124bf20a245c21e715441198be13aae97afb1c956f89044c85a0cdd666c8f8e`
- report: `863beac1387d30809e220e0a05f525e65e1186b4147c51e2d3cea2a306a12789`
