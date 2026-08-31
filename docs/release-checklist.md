# Attune Cadence release checklist

This checklist applies to a model-weight release. A working demo or an
internally good score is not sufficient.

## Candidate selection

1. Record the Git commit, dataset-manifest hashes, training configuration,
   seed, checkpoint hash, hardware, duration, and parameter scope for every
   branch candidate. Generate `lineage.json` with
   `scripts/build_training_lineage.py`; if source files changed during a run,
   bind the record to the start revision and state the reason explicitly.
2. Require each enabled event route to pass its own positive and OOD gates
   through the final exported graph. Do not reuse a frozen-encoder score as an
   ONNX result.
3. Keep styles disabled. A future style head must pass speech-specific positive,
   ordinary-speech, gain, accent, and hostile-lexical gates before composition.
4. Require the affect branch to pass core development, acoustic-preference,
   RAVDESS, BERSt development/confirmation, selective-risk, prediction-share,
   and checkpoint-scope gates.
5. Reject a branch without changing thresholds after seeing its external
   result. Keep the previously retained head when a branch fails.

## Composition and full precision

1. Lock the Cadence graph, calibration, and enabled NumPy probe artifacts by
   SHA-256 before running release evaluation.
2. Export ONNX and retain its parity report. The English-query probe embedding
   must match the frozen training representation before quantization.
3. Collect calibration and release-quality scores with batch size one. The
   temporal event head is not padding-invariant for mixed-length ONNX batches;
   see `research/onnx-batch-padding-audit.md`.
4. Rerun ASR, event, affect, OOD, and hostile-lexical gates on the composed
   graph. Assert that styles are empty. Do not infer full-model acceptance from
   isolated branches.
5. Keep DisfluencySpeech test clips unopened; its one shared speaker prevents
   that split from serving as final external evidence.

## INT8 and deployment

1. Quantize only an accepted full-precision graph. Retain the source/output
   sizes, method, excluded nodes, and quantization report.
2. Refit or confirm calibration for the INT8 graph. Rerun overall metrics and
   independent event, affect, negative-control, and external gates. Compare
   event decisions rather than requiring raw INT8 embeddings to equal FP32.
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
4. Build the release checksum ledger from
   `configs/release/v0.1-artifacts.txt`, then build the Hugging Face bundle from
   `configs/release/v0.1-huggingface-files.txt`. Both builders must hash every
   listed source and fail on missing files.
5. Run `scripts/publish_huggingface.py --dry-run`. It must verify all file
   hashes, required destinations, release-gate schema, and gate results.
6. Upload privately first. Compare the remote file list and hashes with the
   local plan before making the repository public or marking the pull request
   ready for review.
7. Public upload additionally requires a completed redistribution review that
   approves the exact pinned SenseVoice revision, training-source set, and
   SHA-256 digest of every bundled ONNX artifact. The current committed record
   is deliberately not approved and cannot be treated as a legal sign-off.
