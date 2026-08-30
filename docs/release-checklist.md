# Attune Cadence release checklist

This checklist applies to a model-weight release. A working demo or an
internally good score is not sufficient.

## Candidate selection

1. Record the Git commit, dataset-manifest hashes, training configuration,
   seed, checkpoint hash, hardware, duration, and parameter scope for every
   branch candidate.
2. Require the event branch to pass its external WESR, strong-label
   regression, weak human-speech development, and speech-control gates.
3. Require the style branch to pass BERSt development/confirmation, WESR,
   British negative-control, checkpoint-scope, and gain-sweep gates.
4. Require the affect branch to pass core development, acoustic-preference,
   RAVDESS, BERSt development/confirmation, selective-risk, prediction-share,
   and checkpoint-scope gates.
5. Reject a branch without changing thresholds after seeing its external
   result. Keep the previously retained head when a branch fails.

## Composition and full precision

1. Compose only branches with `candidate_passes: true`. Pass each acceptance
   report to `scripts/compose_attune_checkpoint.py`; do not use an unbound
   overlay.
2. Inspect `composition.json`. It must contain the base, every overlay and
   acceptance-report SHA-256, the exact changed tensors, and the output hash.
3. Export ONNX and retain its parity report. Refit calibration on development
   data after composition.
4. Rerun overall ASR/event/style/affect/OOD metrics and all external branch
   gates on the composed graph. Do not infer full-model acceptance from the
   isolated branches alone.
5. Keep DisfluencySpeech test clips unopened; its one shared speaker prevents
   that split from serving as final external evidence.

## INT8 and deployment

1. Quantize only an accepted full-precision graph. Retain the source/output
   sizes, method, excluded nodes, and quantization report.
2. Refit calibration for the INT8 graph. Rerun overall metrics and independent
   event, style, affect, gain, negative-control, and external gates.
3. Benchmark batch inference on CPU and record RTF, p50/p95/p99 latency,
   JSON/XML validity, failures, and the exact hardware.
4. Exercise FastAPI batch and WebSocket streaming with the final model and
   calibration. Verify immutable committed output.
5. Compare the final ASR path with the frozen base on the same evaluation
   clips. Confirm the parameter total remains below 300 million.

## Human regression and release evidence

1. Record a fresh, consented human reading of the hostile regression sentence
   in an ordinary voice without intentional vocal events or styles.
2. Run the final INT8 graph and pass the WAV, inference JSON, model, and
   calibration to `scripts/check_hostile_speech_regression.py`.
3. The normalized transcript must be exact with confidence at least 0.90, and
   the result must contain no event or style hallucination. Retain the hashed
   report; do not include the identifiable WAV in the public bundle.
4. Assemble release metrics with `scripts/assemble_release_metrics.py`, then
   run `scripts/check_release_gates.py`. Schema 1.2 must report every FP and
   INT8 external gate plus the human regression as passed.

## Documentation and publication

1. Replace prerelease metrics in the README, model card, Hugging Face card,
   technical report, experiment registry, and dataset card with measured final
   values. Do not retain superseded numbers as current claims.
2. Record known limitations, disabled labels, calibration behavior, intended
   use, prohibited use, all source attributions, and the SenseVoice derivative
   licence terms.
3. Run `make check` and inspect the final Git diff. Keep the pull request in
   draft until the model artifacts and documentation agree.
4. Build the release checksum ledger and the Hugging Face bundle manifest.
   `scripts/build_huggingface_bundle.py` must hash each explicit
   `SOURCE=DESTINATION` mapping.
5. Run `scripts/publish_huggingface.py --dry-run`. It must verify all file
   hashes, required destinations, release-gate schema, and gate results.
6. Upload privately first. Compare the remote file list and hashes with the
   local plan before making the repository public or marking the pull request
   ready for review.
