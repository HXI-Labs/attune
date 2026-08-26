# Model card

## Status

Phase 0 contains no trained Attune model and downloads no checkpoints. The
planned Phase 1 comparison uses SenseVoice-Small (~234M parameters) as the
intended base, Whisper-Small as fallback, and emotion2vec+ as an affect
baseline. The 2026-08-26 review permits downloading their official weights for
internal baseline runs. It does not authorise fine-tuning or public weight
redistribution, does not approve third-party conversions, and is not legal
advice or lawyer sign-off. MSP-Podcast remains pending and may not be used.

The 14-day gate remains in force. Model licence clearance does not authorise
large fine-tuning before the required baseline report and gate decision.

## Third-party model licences

- SenseVoice-Small source code is MIT, but official weights use the
  [FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).
  Internal work and later public artefacts must attribute
  “SenseVoiceSmall by FunASR/FunAudioLLM,” retain the model name, and link the
  model licence. Derivative weights may remain private; attribution still
  applies when using them. Check GGUF, ONNX, and other conversions separately.
- Whisper's upstream repository applies MIT to its code and original weights.
  The `openai/whisper-small` Hugging Face card currently lists Apache-2.0; this
  project records both and prefers the upstream MIT licence.
- emotion2vec+ seed, base, and large cards identify their weights as
  `other` / `model-license` in the FunASR model-agreement family. The
  emotion2vec repository's MIT/Apache code licensing does not cover these
  weights. Use carries the same attribution duty.

The dated records and source links are in `data/provenance/`.

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
