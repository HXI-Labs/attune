# Model card

## Status

Phase 0 contains no trained Attune model and downloads no checkpoints. The
planned Phase 1 comparison uses SenseVoice-Small (~234M parameters) as the
intended base and Whisper-Small as fallback. SenseVoice's separate model licence
must be reviewed before use or redistribution.

## Intended task

Produce word-timed transcription plus calibrated observable vocal styles,
events, perceived affect, and uncertainty for 0.5–30 second, 16 kHz mono audio.
Primary deployment is local or low-cost near-real-time inference. ONNX and INT8
are later evaluation targets, not current claims.

## Out of scope

Internal-emotion inference, diagnosis, deception detection, protected-trait
inference, speaker identification, and automated high-stakes decisions are out
of scope and prohibited.

## Required evaluation before release

Report ASR error, timing error, event/style metrics, soft-label metrics,
calibration and abstention curves, OOD behaviour, latency/memory, and failures
by language, speaker, acoustic quality, and available demographic slices.
Compare full precision and quantized results. Publish thresholds and their
selection procedure; do not hide selective abstention failures.

## Known limitations

Vocal perception is culturally and contextually variable. Short clips, acted
speech, sarcasm, language mismatch, overlap, clipping, low SNR, far-field audio,
and atypical voices may produce confident errors. Transcript semantics can bias
affect judgments. These limitations require calibration and user-facing caveats
rather than stronger psychological claims.
