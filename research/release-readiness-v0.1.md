# Release readiness status

As of 31 August 2026, Attune Cadence is not yet release-ready. The release
pipeline and deployment artifacts are implemented, but the current candidates
do not pass the predeclared quality gates.

## Completed

- The full test suite passes: 407 tests, with 4 environment-dependent skips.
- Ruff checks and diff validation pass.
- Full-precision ONNX exports, INT8 tooling, calibration, API serving, and
  deterministic JSON/XML rendering are implemented.
- The affect and event training runs completed with source lineage recorded.
- A batch-padding audit found temporal event padding sensitivity; release
  quality scoring therefore remains batch size one.
- The fresh hostile-speech recording used during debugging was processed in
  memory and was not retained.

## Current quality blockers

The affect candidate fails its BERSt development gate (macro-F1 0.2670 versus
the 0.30 minimum). The weak-event candidate fails temporal localization gates
(WESR F1 0.0235 and opened-regression segment F1 0.0345). The earlier style
candidate was rejected because its no-shout false-positive rate was 0.2292.
These thresholds have not been relaxed.

## Next run

Style v0.2 applies the corrected target-specific adaptation path. It is allowed
to update only the style heads and the declared upper encoder parameters. The
run requires a CUDA or supported Apple Silicon environment. Its protocol is in
[`berst-style-v0.2-protocol.md`](berst-style-v0.2-protocol.md).

Only after a candidate passes development, sealed, gain, ASR, calibration,
quantization, and service checks should weights be uploaded to Hugging Face.
The final human hostile-speech regression must be collected again with fresh
consent before release; no prior user audio is retained for that purpose.
