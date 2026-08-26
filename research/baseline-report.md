# Phase 1 baseline report

**Status: harness ready; licensed-data and local-weight evaluation not started.
This document is a gate, not a claim of scientific results.**

No large fine-tuning may begin until this report is completed and reviewed.

## How to fill this report

1. Review and record model licences in `data/provenance/`. SenseVoice-Small has
   a separate weight licence and must not be downloaded before that review.
2. Obtain approved evaluation data through its authorised channel. Do not
   download MSP-Podcast while its ledger remains `licence_review_status:
   pending`; never commit restricted audio.
3. Construct and record speaker/session-disjoint manifests and hashes. Replace
   the synthetic fixture path with approved manifests only in a separate,
   privacy-reviewed evaluation workflow.
4. Place reviewed model weights in local storage or an existing Hugging Face
   cache. The harness never downloads weights. Configure paths as described in
   `docs/baseline-runners.md`.
5. Verify the offline wiring run:

   ```bash
   uv run python scripts/evaluate.py --output research/baseline-results.json
   ```

   The transcript-only runner always executes. Optional runners and modular
   cascades execute only when their local weights and runtimes are available;
   otherwise the JSON report records an actionable skip reason.
6. Add approved-data evaluation reports and fill every section below,
   including slices, runtime hardware, calibration, abstention, OOD behaviour,
   privacy-safe errors, and checkpoint provenance. Synthetic fixture numbers
   must not be substituted for those results.
7. Obtain human review of the completed gate before any large fine-tuning.

## Reproducibility

- Commit/config/run IDs:
- Dataset cards, approved licences, and manifest hashes:
- Speaker/session-disjoint split construction:
- Hardware, software lockfile, seeds, and runtime:
- SenseVoice-Small and Whisper-Small checkpoint provenance:

## Results

- Transcript quality by language and acoustic slice:
- Word/event/style timing:
- Event and style precision/recall/F1, including overlap:
- Affect soft-label metrics:
- ECE, Brier score, reliability plots, and threshold selection:
- Abstention coverage/risk and OOD results:
- Latency, memory, throughput, and local deployment feasibility:
- Speaker, language, recording-quality, and permitted subgroup slices:

## Error analysis

Link representative, privacy-safe errors covering label ambiguity, lexical
bias, short clips, overlap, clipping, low SNR, far field, language mismatch,
atypical voices, and confident failures.

## Gate decision

- Is each ontology label learnable and annotator-supported?
- Is calibration adequate for user-facing language?
- Does a frozen probe suffice?
- Is joint or large fine-tuning supported by evidence?
- What is explicitly deferred or rejected?

## Remaining work before the gate can pass

- Human licence review and exact checkpoint provenance for Whisper-Small,
  SenseVoice-Small, emotion2vec+, and every evaluation corpus.
- Approved, speaker/session-disjoint evaluation manifests and dataset cards.
- Local-weight runs for each adapter and the modular ASR + affect + stub-event
  cascade; the stub event head is not an evaluated event detector.
- Real event/style gold labels, alignment metrics, calibration and abstention
  analysis, OOD and permitted subgroup slices, runtime/memory measurements,
  and privacy-safe qualitative errors.
- A documented gate decision. Until these are complete, the 14-day no-large-
  fine-tuning restriction remains in force.

## Appendix A — synthetic fixture wiring result

These six generated-tone fixtures test only code paths and APS. They are not
recordings of affect, not annotator-supported labels, and not scientific model
results. Full machine-readable output is in `research/baseline-results.json`.

Transcript-only lexicon baseline:

- Schema validity: 6/6 valid.
- ASR: WER `0.0`, CER `0.0` because fixture transcripts are supplied directly.
- Affect: macro-F1 `0.05`, Brier score `0.92027`, soft-target cross-entropy
  `2.65209`, ECE `0.53`.
- APS on five conflict items: acoustic accuracy `0.0` minus lexical accuracy
  `1.0` = **`-1.0`**, correctly exposing this baseline's lexical dependence.
- Event/style fixture support: zero; corresponding F1/IoU values are not
  evidence of event/style performance.
- Whisper-Small, SenseVoice-Small, emotion2vec+, and both cascades skipped
  because no reviewed local weights/runtimes were present. No download was
  attempted.
