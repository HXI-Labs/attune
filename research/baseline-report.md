# Phase 1 baseline report

**Status: not started. This document is a gate, not a claim of results.**

No large fine-tuning may begin until this report is completed and reviewed.

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
