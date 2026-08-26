# Phase 1 baseline report

**Status: fixture smoke-test complete; licensed, speaker-disjoint gold
evaluation not started. This document is a gate, not a claim of scientific
results.**

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
- Real event/style gold labels, alignment metrics, calibration and abstention
  analysis, OOD and permitted subgroup slices, runtime/memory measurements,
  and privacy-safe qualitative errors.
- A documented gate decision. Until these are complete, the 14-day no-large-
  fine-tuning restriction remains in force.

## Fixture smoke-test (not a scientific baseline)

Run at `2026-08-26T19:43:15Z` for Jerry Buaba / HXI Labs from run commit
`1f201b94ed4084bf9081d0786922ceaedd06db20` (starting `main` commit
`04638af61b915fb5d5d87b1409c7ef78fb2c3cb4`). Seed `0` was set for Python,
PyTorch, and Python hash randomisation. The one-time official checkpoint fetch
used a cache under `/tmp`, outside the Git tree. Evaluation then ran with
Hugging Face, Transformers, ModelScope, and FunASR offline flags enabled.

The six fixtures contain generated tones/noise, not human speech or natural
affect. The numbers below prove only that each adapter and cascade loaded local
weights, ran end-to-end on CPU, emitted schema-valid output, and reported
runtime. They are neither model-quality evidence nor the sealed gold baseline.

Hardware was a four-vCPU Intel Xeon VM with 15 GiB RAM and no available GPU.
Software was Python 3.12.3, PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130,
Transformers 5.16.1, FunASR 1.4.4, and Pydantic 2.13.4.

| Runner | Schema valid | RTF | Mean latency per 0.5 s clip |
|---|---:|---:|---:|
| transcript-only-lexicon | 6/6 | 0.00024 | 0.12 ms |
| Whisper-Small | 6/6 | 3.96186 | 1,980.93 ms |
| SenseVoiceSmall | 6/6 | 3.46482 | 1,732.41 ms |
| emotion2vec+ base | 6/6 | 1.54092 | 770.46 ms |
| Whisper-Small + emotion2vec+ + stub event head | 6/6 | 5.38111 | 2,690.55 ms |
| SenseVoiceSmall + emotion2vec+ + stub event head | 6/6 | 4.67179 | 2,335.89 ms |

These timings include model construction and loading inside every `predict()`
call; they are wiring measurements, not optimised serving benchmarks.
First-result latency was not measured. No runners were skipped in the completed
run. The first attempt stopped before SenseVoiceSmall inference with
`ModuleNotFoundError: No module named 'torchaudio'`; after torchaudio 2.11.0 was
installed in the local environment and import compatibility was verified, the
full offline rerun completed. The machine-readable report records this resolved
failure rather than hiding it.

Official weights used:

- OpenAI **Whisper-Small** (`openai/whisper-small`), revision
  `973afd24965f72e36ca33b3055d56a652f456b4d`; upstream Whisper is MIT licensed.
- **SenseVoiceSmall by FunASR/FunAudioLLM**, revision
  `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`, under the
  [FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).
- **emotion2vec+ base by emotion2vec and FunASR/FunAudioLLM**, revision
  `b318240bfe67db81a8c572ecb37ce9c3759b81c9`; the base model name and variant
  are retained, with the [FunASR model licence](https://github.com/alibaba-damo-academy/FunASR)
  attribution.

Exact weight-file SHA-256 values, full runtime precision, software metadata,
skip reasons, and the resolved failure are in
`research/fixture-smoke-test.json`. No fine-tuning occurred, no speech corpus
was downloaded, and no weights or caches are part of this change.

Gold evaluation remains blocked on a licence-approved, speaker-disjoint
labelled clip set with real speech and event/style/affect annotations. The
planned 100–200 clip inspection coverage and unapproved candidate sources are
listed in `research/inspection-set.md`; every candidate remains
`licence_review_status: pending`.

## Inspection-set smoke (weak/acted labels, not gold)

This first real-speech inspection ran at `2026-08-26T20:43:30Z` from commit
`b936e51c51dac8c4d88eca00d61a273007819392`. It used all 150 checksum-verified
manifest clips: 80 VocalSound event clips and 70 CREMA-D speech clips, totalling
503.311 seconds. These source labels are **weak and acted**. This is not the
sealed gold baseline, does not pass the gate decision, and is not evidence about
speakers' internal emotional states. No fine-tuning occurred.

The run used four logical Intel Xeon CPU cores, 15.64 GiB RAM, and no GPU on
Linux 6.12.94+ x86-64. The runtime was Python 3.12.3, PyTorch 2.13.0,
torchaudio 2.11.0, Transformers 5.16.1, FunASR 1.4.4, and Pydantic 2.13.4.
Checkpoint files lived under `/tmp`, outside the Git tree. The actual evaluation
ran with Hugging Face, Transformers, ModelScope, and FunASR offline flags set.

| Runner | Schema valid | CREMA-D WER | VocalSound event macro-F1 | CREMA-D affect macro-F1 | APS | RTF | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| transcript-only lexicon | 70/70 | 0.0000 supplied | N/A | 0.0833 | -1.0000 | 0.00003 | 0.08 ms |
| Whisper-Small | 150/150 | 0.1226 | 0.0000 | 0.0833 | -1.0000 | 0.5145 | 1,726.48 ms |
| SenseVoiceSmall | 150/150 | 0.0806 | 0.0000 | 0.0833 | -1.0000 | 0.0390 | 130.80 ms |
| emotion2vec+ base | 150/150 | N/A | 0.0000 | 0.9208 | 0.8667 | 0.0377 | 126.49 ms |
| Whisper + emotion2vec+ + stub event | 150/150 | 0.1226 | 0.0000 | 0.9208 | 0.8667 | 0.4947 | 1,659.79 ms |
| SenseVoice + emotion2vec+ + stub event | 150/150 | 0.0742 | 0.0000 | 0.9208 | 0.8667 | 0.0662 | 222.16 ms |

WER and CER use lowercase alphanumeric normalization and only the 70 CREMA-D
clips with source transcripts. Transcript-only WER is zero by construction
because that runner receives the supplied source transcript; emotion2vec+ is
affect-only, so its retained transcript hints are not scored as ASR. APS is
computed on 60 acoustic-versus-lexical conflict clips. CREMA-D source emotion
codes map as `HAP → joy`, `SAD → distress`, and `NEU → neutral`. The high
emotion2vec+ agreement is against acted source labels only and must not be
presented as gold performance.

All audio-capable runners emitted empty event lists because the current adapters
and modular cascades use the explicit stub event head. Their zero VocalSound
event F1 and position-aware scores are real negative results against 80 weak
whole-clip labels, not missing or invented metrics. Transcript-only has no
VocalSound transcript input, so its 80 event clips are marked not applicable.
All six runners completed without a model or clip failure. Full precision,
per-class support, skip/failure fields, runtime, and privacy-safe examples are
in `research/inspection-smoke-results.json`.

### Inspection error notes

- `1007_IEO_SAD_HI.wav`: Whisper produced “So let's go to class” for the acted
  “It's eleven o'clock” line (five normalized word edits).
- `1001_IWL_HAP_XX.wav`: SenseVoice produced “i would like to do an alarm
  clock” (three normalized word edits).
- `1001_IEO_SAD_HI.wav`: emotion2vec+ predicted `joy` against the weak acted
  `distress` source label.
- `f0418_0_cough.wav`: every audio-capable runner emitted no event against the
  weak `cough` source label; this reflects the stub event integration.

No raw audio is included in these notes or in the Git diff.

### Attribution, coverage gaps, and gate boundary

VocalSound by Gong, Yu, and Glass (ICASSP 2022) is used under CC BY-SA 4.0.
CREMA-D by Cao et al. (2014) is attributed under ODbL 1.0 for the database and
DbCL 1.0 for individual contents. The run used OpenAI Whisper-Small (upstream
Whisper MIT), SenseVoiceSmall by FunASR/FunAudioLLM under the FunASR Model Open
Source License Agreement v1.1, and emotion2vec+ base by emotion2vec and
FunASR/FunAudioLLM under the documented FunASR model licence. Exact checkpoint
revisions and hashes remain recorded in `research/fixture-smoke-test.json` and
the corresponding `data/provenance/` entries.

Still missing are Ghanaian English, verified British English, cry/sob, and
verified shout/whisper slices, as well as spontaneous affect and inline events
within speech. MSP-Podcast, RAVDESS, Ghana English ASR, SAVEE, DEED, and EmoV-DB
were not downloaded. Gold review, calibration, abstention/OOD analysis, missing
slice coverage, and an explicit human gate decision remain outstanding; the
gate is not passed.
