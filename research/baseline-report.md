# Phase 1 baseline report

**Status: Phase 1 baseline package complete. The scientific gold gate remains
closed. Source/acted labels are weak, and no real inspection row has been
promoted to gold. See `research/phase1-close.md`.**

## Executive result

The inspected cascade combines SenseVoice-Small transcript/AED,
emotion2vec+ affect, and frozen-SenseVoice VocalSound/FSD50K linear probes.
Validation-selected `none` logits suppress out-of-domain probe emissions.
JSON is authoritative and XML is a deterministic projection.

| 310-clip slice | WER | Affect macro-F1 | Cascade target macro-F1 | OOD FPR | All-prediction micro-F1 |
|---|---:|---:|---:|---:|---:|
| Original 150 | 0.0703 | 0.9208 | 0.7839 | 0.0682 | 0.7039 |
| Licence-clean expansion 160 | 0.1900 | 0.8679 | 0.7272 | 0.0227 | 0.6927 |

These are inspection results against weak source/acted targets. The confidence
calibration and DCASE timing diagnostic added at closure are reported below;
neither changes the label status.

## Phase 1 gate checklist

| Requirement | Evidence | Status |
|---|---|---|
| ASR | CREMA-D, British Common Voice, Ghanaian-English reports | measured |
| timing/alignment | no fabricated word timing; clip probes remain utterance-scoped; separate DCASE strong-label diagnostic | bounded |
| behaviour labels | VocalSound, CREMA-D, FSD50K, combined cascade | measured on weak/acted labels |
| calibration | validation-only temperature scaling; test ECE/Brier | measured, not gold |
| abstention | validation-selected `none` logits and 310-clip OOD rerun | measured |
| OOD/subgroups | cross-probe OOD, British English, Ghanaian English, licence-clean expansion | bounded |
| runtime | per-run elapsed/audio RTF where available | measured on CPU |
| error analysis | `research/error-analysis/` confusion, OOD, runtime, and qualitative notes | complete |
| gold review | strict append-only ledger contract and fixture dry-run | protocol only |

## Reproducibility

- Environment and package versions are pinned by `uv.lock`.
- Dataset/model reviews and exact revisions are under `data/provenance/`.
- Every bounded audio selection has a committed hashed manifest; audio, model
  weights, embeddings, and `.pt` heads are gitignored.
- VocalSound is speaker-disjoint; FSD50K is clip-disjoint because it has no
  speaker identity; British Common Voice uses 100 distinct client IDs; Ghana
  exposes no speaker IDs and is marked accordingly.
- SenseVoice extraction is direct frontend/encoder, dither `0.0`, frozen
  parameters, and a representation-specific cache fingerprint.
- Machine-readable results are under `research/*.json` and
  `research/error-analysis/*.json`.

## Remaining work before the gate can pass

- Human review and adjudication of the existing 310 clips; no current row is gold.
- Natural, speech-overlapping onset/offset labels beyond the synthetic DCASE diagnostic.
- Coverage for `crying_speech` and broader in-the-wild behaviour.
- Better Ghanaian-English WER evidence on a licence-compatible, speaker-traceable set.
- Broader open-world OOD, recording-quality, and demographic evaluation.
- A separate reviewed decision before any scientific gold claim.

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

This real-speech inspection rerun completed at `2026-08-26T21:12:19Z` from
merged `main` commit `a2ad5f0fc3ef0cf7055fa854fddf0f550ebe243f`. It used all
150 checksum-verified
manifest clips: 80 VocalSound event clips and 70 CREMA-D speech clips, totalling
503.311 seconds. These source labels are **weak and acted**. This is not the
sealed gold baseline, does not pass the gate decision, and is not evidence about
speakers' internal emotional states. No fine-tuning occurred.

The run used four logical Intel Xeon CPU cores, 15.64 GiB RAM, and no GPU on
Linux 6.12.94+ x86-64. The runtime was Python 3.12.3, PyTorch 2.13.0,
torchaudio 2.11.0, Transformers 5.16.1, FunASR 1.4.4, and Pydantic 2.13.4.
Checkpoint files lived under `/tmp`, outside the Git tree. The actual evaluation
ran with Hugging Face, Transformers, ModelScope, and FunASR offline flags set.

Baseline B replaces the SenseVoice stub event path with SenseVoice-Small's
off-the-shelf rich-transcription/AED tags. It performs no training or
fine-tuning. Known tags are conservatively mapped to the existing event
ontology, and SenseVoice events/styles are retained by the SenseVoice cascade.
The official off-the-shelf AED inventory covers laughter, crying, coughing,
sneezing, and breath, plus non-Attune BGM and applause. It does **not** cover
`sigh` or `throat_clear`. The parser accepts those names as forward-compatible
ontology aliases, but that does not make them outputs of the released model.
Because these tags do not provide frame boundaries, predictions are provisional
whole-clip spans with confidence `0.0` when no score is exposed. Events remain
**utterance-level, not localized**; the temporal IoU and position-aware results
only compare whole-clip spans and are not evidence of frame-level localization.

| Runner | Schema valid | CREMA-D WER | VocalSound event macro-F1 | CREMA-D affect macro-F1 | APS | RTF | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| transcript-only lexicon | 70/70 | 0.0000 supplied | N/A | 0.0833 | -1.0000 | 0.00003 | 0.08 ms |
| Whisper-Small | 150/150 | 0.1226 | 0.0000 | 0.0833 | -1.0000 | 0.4984 | 1,672.40 ms |
| SenseVoiceSmall | 150/150 | 0.0806 | 0.3213 | 0.0833 | -1.0000 | 0.0397 | 133.07 ms |
| emotion2vec+ base | 150/150 | N/A | 0.0000 | 0.9208 | 0.8667 | 0.0325 | 109.20 ms |
| Whisper + emotion2vec+ + ASR event output | 150/150 | 0.1226 | 0.0000 | 0.9208 | 0.8667 | 0.4904 | 1,645.50 ms |
| SenseVoice + emotion2vec+ + ASR event output | 150/150 | 0.0742 | 0.3213 | 0.9208 | 0.8667 | 0.0664 | 222.69 ms |

SenseVoiceSmall and the SenseVoice cascade both improved from the previous stub
event macro-F1 of **0.0000** to **0.3213**. The cascade retains SenseVoice AED
events, so its event scores are identical:

| Event class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| laugh | 16 | 0.8750 | 0.8750 | 0.8750 |
| sob | 0 | 0.0000 | 0.0000 | 0.0000 |
| sigh | 16 | 0.0000 | 0.0000 | 0.0000 |
| cough | 16 | 0.4815 | 0.8125 | 0.6047 |
| throat_clear | 16 | 0.0000 | 0.0000 | 0.0000 |
| sneeze | 16 | 1.0000 | 0.6250 | 0.7692 |
| breath | 0 | 0.0000 | 0.0000 | 0.0000 |

There were 37 matches, 80 weak references, and 65 predictions. Whole-clip
temporal IoU was 1.0000 for matched predictions and the position-aware score was
0.4625; both are artifacts of utterance-level spans rather than localization
quality.

WER and CER use lowercase alphanumeric normalization and only the 70 CREMA-D
clips with source transcripts. Transcript-only WER is zero by construction
because that runner receives the supplied source transcript; emotion2vec+ is
affect-only, so its retained transcript hints are not scored as ASR. APS is
computed on 60 acoustic-versus-lexical conflict clips. CREMA-D source emotion
codes map as `HAP → joy`, `SAD → distress`, and `NEU → neutral`. The high
emotion2vec+ agreement is against acted source labels only and must not be
presented as gold performance.

Whisper-Small, emotion2vec+, and the Whisper cascade still emit no events, so
their zero VocalSound event F1 values remain real negative results against 80
weak whole-clip labels. Transcript-only has no VocalSound transcript input, so
its 80 event clips are marked not applicable. All six runners completed without
a model or clip failure. Full precision, per-class support, skip/failure fields,
runtime, and privacy-safe examples are in
`research/inspection-smoke-results.json`.

### Inspection error notes

- `1007_IEO_SAD_HI.wav`: Whisper produced “So let's go to class” for the acted
  “It's eleven o'clock” line (five normalized word edits).
- `1001_IWL_HAP_XX.wav`: SenseVoice produced “i would like to do an alarm
  clock” (three normalized word edits).
- `1001_IEO_SAD_HI.wav`: emotion2vec+ predicted `joy` against the weak acted
  `distress` source label.
- SenseVoice detected 37 of 80 weak event references. It detected no `sigh` or
  `throat_clear` references because neither class exists in the released
  SenseVoice AED inventory. Their F1 of 0 is an expected model-coverage gap,
  not evidence of a parser defect. `cough` precision of 0.4815 indicates
  additional utterance-level cough predictions outside the weak cough class.

No raw audio is included in these notes or in the Git diff.

### Attribution, coverage gaps, and gate boundary

VocalSound by Gong, Yu, and Glass (ICASSP 2022) is used under CC BY-SA 4.0.
CREMA-D by Cao et al. (2014) is attributed under ODbL 1.0 for the database and
DbCL 1.0 for individual contents. The run used OpenAI Whisper-Small (upstream
Whisper MIT), SenseVoiceSmall by FunASR/FunAudioLLM under the FunASR Model Open
Source License Agreement v1.1, and emotion2vec+ base by emotion2vec and
FunASR/FunAudioLLM under the documented FunASR model licence. Exact checkpoint
revisions and hashes are recorded in `research/fixture-smoke-test.json`;
licence constraints and attribution are recorded in `data/provenance/`.

Still missing are verified British English, cry/sob, and verified
shout/whisper slices, as well as spontaneous affect and inline events within
speech. MSP-Podcast, RAVDESS, SAVEE, DEED, and EmoV-DB were not downloaded.
Gold review, calibration, abstention/OOD analysis, missing slice coverage, and
an explicit human gate decision remain outstanding; the gate remains closed.

## Ghanaian-English WER slice — NC research-only

This **NC research-only** run used code at
`acf27aea44a3d3cae1dc176d5c4ece1900572633` and 100 streamed clips from
`ghananlpcommunity/ghana-english-asr-2700hrs` revision
`893a08082ec0f34b5d2fbec56f1ab2230ebea1e7`. The 1,328.911 seconds of audio
were converted to mono 16 kHz PCM16 WAV and kept under gitignored `data/raw/`.
Only the JSONL manifest, transcripts, hashes, and reproducible fetch path are
committed. No full shard or full-corpus download occurred.

The corpus exposes only `audio`, `corrected_text`, and `duration_ss`; it has no
speaker identifier. Consequently this deterministic first-100-valid-row slice
cannot be speaker-disjoint, may contain repeated speakers or adjacent broadcast
segments, and cannot support speaker-leakage checks. It is an accent/domain
inspection slice, not a population-representative Ghanaian-English benchmark.

| Runner | Licence scope | Clips | Ghanaian-English WER | CER | CREMA-D acted US English WER | Absolute WER difference |
|---|---|---:|---:|---:|---:|---:|
| SenseVoiceSmall | NC research-only | 100 | 0.2365 | 0.1768 | 0.0806 | +0.1559 |
| Whisper-Small | NC research-only | 100 | 0.2562 | 0.2099 | 0.1226 | +0.1336 |

WER and CER use the same lowercase alphanumeric normalization as the CREMA-D
inspection. Both runners completed all 100 clips without failure, offline after
the one-time dataset and reviewed-weight fetch. Whisper was run in known-English
transcription mode; this fixes an adapter defect where `language_hint="en"` was
previously ignored and incorrect language detection could collapse a valid
English clip to one token. On four CPU cores, SenseVoiceSmall had RTF 0.0193 and
Whisper-Small RTF 0.0918. CREMA-D is acted US English rather than a matched
domain control, so the differences are descriptive and combine accent, domain,
recording, and transcript effects.

The source is attributed to the Ghana NLP Community under CC BY-NC 4.0.
Whisper-Small is attributed to OpenAI under the upstream Whisper MIT licence.
SenseVoiceSmall is attributed to FunASR/FunAudioLLM under the FunASR Model Open
Source License Agreement v1.1. Exact checkpoint revisions and hashes are in
`research/ghana-english-wer-results.json`.

This manifest and every metric in this section are **NC research-only**. They
must never be used for a commercially redistributed training set and are not
covered by the repository's MIT code licence. No fine-tuning occurred, this
slice does not pass the gate, and the gate remains closed.

## Stage 2 frozen VocalSound event probe

The Stage 2 run used code at `4a3e8b9`, seed 0, and one checksum-verified
VocalSound 16 kHz WebDataset training shard (`wds-audio-train-000000.tar`,
MD5 `003d0a4cb29afa422042043aaec01c41`) from Zenodo record 14650192. VocalSound
remains attributed to Gong, Yu, and Glass and governed by CC BY-SA 4.0. The
source labels remain five-way acted VocalSound events, not reviewed gold.

The frozen, parameter-free `fixed-logmel-v1` embedding had zero trainable
parameters; only the 2,005-parameter linear head was trained. The sampled pool
contained 646 clips from 576 speakers. A speaker-level split produced 514
training clips/461 speakers and 132 validation clips/115 speakers. All 16
speakers appearing in either inspection-manifest partition were excluded from
both pools. The test set was the 80 inspection VocalSound clips from those 16
speakers, balanced at 16 clips per class.

| Event | Probe test F1 | SenseVoiceSmall F1 |
|---|---:|---:|
| laugh | 0.4118 | 0.8750 |
| sigh | 0.4516 | 0.0000 |
| cough | 0.2963 | 0.6047 |
| throat_clear | 0.4444 | 0.0000 |
| sneeze | 0.7500 | 0.7692 |

Probe test accuracy was 0.4750 and five-class macro-F1 was 0.4708. The
SenseVoice numbers are the existing off-the-shelf inspection results, not a
fine-tuned comparator; in particular, its released AED inventory has no `sigh`
or `throat_clear` class. Full-precision metrics, split membership, label counts,
and loss history are in `research/stage2-vocalsound-probe-metrics.json`.

Reproduce on CPU:

```bash
uv sync --extra torch
uv run python scripts/prepare_dataset.py --download
curl -fL -o data/raw/vocalsound-16k/wds-audio-train-000000.tar \
  https://zenodo.org/api/records/14650192/files/wds-audio-train-000000.tar/content
echo "003d0a4cb29afa422042043aaec01c41  data/raw/vocalsound-16k/wds-audio-train-000000.tar" | md5sum -c -
tar -xf data/raw/vocalsound-16k/wds-audio-train-000000.tar \
  -C data/raw/vocalsound-16k --wildcards "*.wav"
uv run python scripts/train_probe.py \
  --metrics-output research/stage2-vocalsound-probe-metrics.json
```

The run used Python 3.12.3, PyTorch 2.13.0+cpu, four logical CPUs, and no GPU.
The generated head checkpoint stayed under gitignored `artifacts/` and is not
included. This limited acted-event probe does not pass the gate.

## H1: frozen SenseVoiceSmall encoder probe

The H1 run used code at `1d0db85`, seed 0, the same checksum-verified
`wds-audio-train-000000.tar` pool, the same deterministic speaker split, and
the same 80 inspection clips as the committed log-mel probe. The train,
validation, and test speaker lists match the committed report exactly: 514
clips/461 speakers for training, 132 clips/115 speakers for validation, and 80
clips/16 speakers for inspection testing. Every inspection-manifest VocalSound
speaker was excluded from training and validation.

Official SenseVoiceSmall revision
`3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` was loaded from a local checkpoint
whose `model.pt` SHA-256 is
`833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea`.
All 233,999,167 model parameters were set to `requires_grad=False`, the model
ran under inference mode, and no model parameter was passed to an optimizer.
The four prepended rich-transcription query frames were excluded. Eight
temporal means plus acoustic-frame mean and standard deviation were pooled from
the remaining 512-dimensional encoder frames. FunASR frontend dither was set to
`0.0` to remove inference-time randomness. Only a 25,605-parameter linear
five-class head was trained. Fresh extraction and a cached rerun produced
identical training history and metrics (excluding cache hit/miss counters).

| Event | Frozen SenseVoice encoder | Frozen log-mel | Off-the-shelf SenseVoice AED |
|---|---:|---:|---:|
| laugh | 0.8571 | 0.4118 | 0.8750 |
| sigh | 0.9032 | 0.4516 | 0.0000 |
| cough | 0.8387 | 0.2963 | 0.6047 |
| throat_clear | 0.7586 | 0.4444 | 0.0000 |
| sneeze | 0.9412 | 0.7500 | 0.7692 |
| **Macro-F1** | **0.8598** | **0.4708** | **0.3213** |

The frozen encoder probe reached 0.8625 accuracy and 0.8598 five-class
macro-F1. It improved macro-F1 by 0.3890 over the committed log-mel probe and
by 0.5385 over the existing AED event score. In particular, `sigh` improved
from 0.4516/0.0000 to 0.9032 and `throat_clear` improved from 0.4444/0.0000 to
0.7586 relative to log-mel/AED. On this fixed weak-label inspection protocol,
these results support H1: the frozen encoder exposes event-discriminative
information that the released AED tag inventory does not expose.

This is evidence on standalone, crowdsourced acted VocalSound events only. It
does not establish frame localization, natural inline-event performance,
calibration, abstention, OOD robustness, or performance on reviewed gold data.
The gate remains closed. Full-precision metrics, loss history, checkpoint
provenance, freeze evidence, and exact split membership are in
`research/sensevoice-frozen-probe-metrics.json`. Audio, cached embeddings,
weights, and the trained head remain gitignored.

The run used Python 3.12.3, PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130,
FunASR 1.4.4, four CPU threads, no available GPU, and early stopping after eight
epochs. SenseVoiceSmall by FunASR/FunAudioLLM is used under the
[FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).
VocalSound by Gong, Yu, and Glass is used under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## British-English WER slice — Mozilla Common Voice CC0

This run used code at `84c511d24c105d3b9ffd51d5c20ac68a40913335`
and 100 checksum-verified Mozilla Common Voice Corpus 17.0 English clips from
the pinned `fixie-ai/common_voice_17_0` transport revision
`34f78a43893414e7b6e271ba94c1d5e05f18b239`. The exact self-declared `accent`
values represented are **`England English`** (96 clips) and
**`Scottish English`** (4 clips). The configured filter also accepts
`Welsh English`, but no Welsh row was selected before reaching 100 speakers.
These labels are self-reported metadata, not independent verification of
nationality or residence.

The deterministic selector streamed only metadata through source row 1,798,
then fetched only the 100 selected audio assets. It did not download the full
Common Voice corpus or a full Parquet shard. Each selected row has a distinct
hashed `client_id`, so the slice contains 100 clips from 100 speakers. The
550.030 seconds of audio were converted to mono 16 kHz PCM16 WAV and kept under
gitignored `data/raw/`; no audio is committed.

| Runner | British-English WER | CER | CREMA-D acted US WER | Difference vs CREMA-D | Ghanaian English NC WER | Difference vs Ghana |
|---|---:|---:|---:|---:|---:|---:|
| SenseVoiceSmall | 0.1149 | 0.0478 | 0.0806 | +0.0343 | 0.2365 | -0.1216 |
| Whisper-Small | 0.0997 | 0.0378 | 0.1226 | -0.0229 | 0.2562 | -0.1565 |

Both runners completed 100/100 clips without failure after a one-time official
checkpoint fetch. Evaluation then ran with Hugging Face, Transformers,
ModelScope, and FunASR offline flags enabled. On four CPU cores,
Whisper-Small had RTF 0.1625 and SenseVoiceSmall RTF 0.0305. The four Scottish
clips are too few for a meaningful standalone estimate; their full-precision
descriptive metrics remain in
`research/common-voice-british-wer-results.json`.

WER and CER use the same lowercase alphanumeric normalization as the CREMA-D
and Ghanaian-English slices. The comparisons are descriptive, not controlled:
CREMA-D is acted US speech and the NC Ghanaian slice is broadcast-domain
speech, so accent, speaker, recording, prompt, and transcript differences are
confounded.

Mozilla Common Voice is used under CC0 1.0 with voluntary attribution. OpenAI
Whisper-Small is attributed under the upstream Whisper MIT licence.
SenseVoiceSmall by FunASR/FunAudioLLM is used under the
[FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).
Exact checkpoint revisions and hashes are in the machine-readable result. No
fine-tuning occurred. This accent slice does not evaluate affect, events,
localization, calibration, abstention, or OOD robustness; it does not pass the
gate, and the gate remains closed.

## Licence-clean event and affect expansion

This packed inspection revision adds 100 individually fetched FSD50K clips
(25 each of `Shout`, `Whispering`, `Crying_and_sobbing`, and `Screaming`) plus
60 CREMA-D clips (20 actors × `ANG`, `FEA`, and `DIS`). All audio is mono
16 kHz PCM16 under gitignored `data/raw/`; every selected converted file has a
SHA-256 in `data/manifests/licence-clean-inspection.jsonl`.

FSD50K selection used official ground truth and clip metadata before any audio
fetch. Only individual Freesound clips licensed CC0 or CC BY were eligible:
16 CC0 and 84 CC BY clips were selected, each carrying exactly one of the four
target classes. Uploader attribution is retained per row. CC BY-NC, Sampling+,
and cross-target clips were excluded, and no full FSD50K audio archive was
downloaded. FSD50K curation and annotations remain attributed to Fonseca et al.
under CC BY 4.0.

The weak-label boundary is explicit: FSD50K `Shout` is a standalone sound, not
speech-embedded shouting; `Whispering` maps weakly to `whispering`;
`Crying_and_sobbing` maps to `sob`, not `crying_speech`; and `Screaming`
remains separate. CREMA-D intensity remains source metadata and is never
mapped to shouting or whispering. CREMA-D has no surprise source category.
Its 20 expansion actors are disjoint from the original 70-clip inspection set.

The evaluation script runs reviewed, pinned OpenAI Whisper-Small and
SenseVoiceSmall by FunASR/FunAudioLLM offline after checkpoint fetch. It
reports Whisper-versus-SenseVoice WER/CER and categorical affect on CREMA-D,
SenseVoice AED tags on FSD50K, and can run an already-trained frozen
SenseVoice event-probe head as an OOD diagnostic. It never trains a new head
or fine-tunes an encoder. The probe checkpoint is intentionally gitignored;
if unavailable, the machine report records that step as blocked rather than
retraining.

The offline run completed from commit `618184c` with no model or clip failures.
Both checkpoints match the hashes used by the earlier Common Voice run.

| Runner | CREMA-D clips | WER | CER | Affect accuracy | Affect macro-F1 |
|---|---:|---:|---:|---:|---:|
| Whisper-Small | 60/60 | 0.1967 | 0.2877 | 0.0000 | 0.0000 |
| SenseVoiceSmall | 60/60 | 0.1900 | 0.2561 | 0.0000 | 0.0000 |

Whisper labelled all 60 affect outputs `neutral`. SenseVoice labelled 58
`neutral` and two `distress`; none matched the balanced weak ANG/FEA/DIS
references. These are categorical output checks, not evidence that acted
emotion categories are internal states. The comparison also does not isolate
speaker, sentence, or intensity effects.

SenseVoice AED detected `sob` on 8/25 (0.32) `Crying_and_sobbing` clips. It
emitted no matching `shouting` or `whispering` style tags on the respective
25-clip classes. The 25 `Screaming` clips have no Attune shouting target and
remain a separate reported source class. The released AED inventory and this
adapter do not expose a scream target.

The frozen SenseVoice encoder probe evaluation is explicitly incomplete:
status `blocked_missing_existing_checkpoint`. The trained linear head from the
earlier probe is correctly gitignored and was not present on this VM. No new
head was trained, the encoder was not fine-tuned, and no checkpoint was
committed. Exact outputs, checkpoint hashes, runtime, failures, and the blocked
status are in `research/licence-clean-inspection-results.json`.

RAVDESS, SAVEE, DEED, EmoV-DB, and MSP-Podcast remain gated and were not
downloaded. The existing Ghanaian-English NC research-only slice was not
expanded. Human listening, reviewed gold labels, localization, calibration,
abstention/OOD thresholds, and an approved gate decision remain incomplete.
The gate remains **CLOSED**.

## Phase 1 hard-hole follow-up

### Affect wiring diagnosis

The original 60-clip affect table mixed three different paths without exposing
their provenance. Whisper-Small has no affect head; its `neutral` 60/60 came
from Attune's transcript-lexicon placeholder. SenseVoice affect tags were read,
but the report discarded the raw rich-transcription output. The standalone
emotion2vec+ acoustic SER adapter, which reached 0.9208 macro-F1 on the original
NEU/HAP/SAD inspection set, was not run on the ANG/FEA/DIS expansion at all.

Raw SenseVoice output resolves the apparent collapse. It emitted
`<|EMO_UNKNOWN|>` on 58 clips and `<|SAD|>` on two. The two `SAD` tags mapped
to Attune `distress`; the 58 unmapped SER results fell back to the neutral
transcript lexicon. Thus the 58-neutral/2-distress schema histogram was not an
ASR-path drop of mapped emotion tags, but it also was not evidence that a
working acoustic classifier predicted neutral 58 times. The released
SenseVoice SER behavior on this acted slice is mostly unknown.

The corrected evaluator now runs emotion2vec+ directly and as the affect
component in both Whisper and SenseVoice cascades. All three acoustic-affect
paths produced the same categorical result:

| Affect path | Accuracy | Macro-F1 | Raw/schema prediction counts |
|---|---:|---:|---|
| emotion2vec+ | 0.8500 | 0.8679 | angry 23, fearful 14, disgusted 20, sad 2, happy 1 |
| Whisper + emotion2vec+ | 0.8500 | 0.8679 | identical |
| SenseVoice + emotion2vec+ | 0.8500 | 0.8679 | identical |

Per-class emotion2vec+ F1 was 0.9302 anger, 0.8235 fear, and 0.8500 other.
Its raw `厌恶/disgusted` label occurred 20 times and maps explicitly to Attune
`other`. CREMA-D `DIS` remains `other`, never `distress`; a model without an
`other`/disgust output has structurally zero DIS recall. The machine report
contains raw-label histograms and two raw-to-schema examples for each source
emotion and each runner.

This fixes an evaluation wiring omission and shows that emotion2vec+ did not
collapse on the expansion. It does not turn acted source labels into gold,
establish internal emotional state, or pass the gate. OpenAI Whisper-Small,
SenseVoiceSmall by FunASR/FunAudioLLM, and emotion2vec+ by emotion2vec and
FunASR/FunAudioLLM retain their recorded attributions and model licences.

### Frozen FSD50K coverage probe

The authorised H1 follow-up trained a new 20,484-parameter four-way linear head
on frozen SenseVoiceSmall encoder embeddings. The bounded pool contains 256
training clips (64 per class) and 64 validation clips (16 per class). All 100
inspection rows remain the balanced test set and are excluded before
selection. FSD50K has no speaker IDs, so this is clip-disjoint rather than
speaker-disjoint; uploader attribution is not treated as speaker identity.
Every clip is individually fetched, CC0 or CC BY, and carries exactly one
target class. No NC, Sampling+, cross-target, or full-archive audio is used.

The extraction path is now direct: official FunASR frontend with dither `0.0`,
then the frozen encoder under inference mode. It no longer calls `generate()`
or captures a forward hook. Investigation found that the version-1 hook path's
`generate()` reset restored FunASR's frontend configuration, including dither
`1.0`, after the extractor had set its frontend reference to zero. Version-2
embeddings have a new representation fingerprint; version-1 caches and heads
are not silently reused.

All 233,999,167 SenseVoiceSmall parameters had gradients disabled and zero
encoder parameters entered the optimizer. Only the linear head trained. The
fresh run had 420 cache misses; a second run had 420 cache hits and reproduced
the full loss history and every validation/test metric exactly. The
inspection result was 0.7400 accuracy and 0.7410 four-way macro-F1:

| Source/probe class | Frozen encoder probe F1 | AED detection rate | AED one-vs-rest F1 |
|---|---:|---:|---:|
| shout | 0.7391 | 0.0000 | 0.0000 |
| whisper | 0.8750 | 0.0000 | 0.0000 |
| sob | 0.7500 | 0.3200 | 0.4848 |
| scream | 0.6000 | 0.0000 | 0.0000 |

The requested AED values `0/0/0.32/0` are detection rates, not F1. They remain
reported under that name. From the committed raw AED outputs, sob has eight
true positives, no false positives, and 17 false negatives, hence one-vs-rest
F1 0.4848. `Screaming` stays a separate diagnostic class and is never remapped
to Attune `shouting`; `Crying_and_sobbing` maps to `sob`, not
`crying_speech`.

These results support H1 on this weak, standalone FSD50K slice: frozen encoder
features expose all four distinctions that the released AED outputs miss or
under-cover. They do not establish speech-embedded style detection, reviewed
gold performance, localization, calibration, abstention, or OOD robustness.
No encoder fine-tuning occurred, no weights or probe checkpoint are committed,
and the gate remains **CLOSED**.

## Combined cascade, abstention, and ablations

PR #18 joined the concrete components without changing their ontology
boundaries. SenseVoice transcript/AED runs first; emotion2vec+ replaces the
ASR adapter's placeholder affect; validation-selected VocalSound and FSD50K
probes contribute only when their `none`-margin accepts. Duplicate labels are
suppressed in AED, VocalSound, FSD50K order. `scream` is never collapsed into
`shouting`, and `sob` never implies `crying_speech`.

| Slice / event-style ablation | Target macro-F1 | All-prediction micro-F1 |
|---|---:|---:|
| Original 150: AED only | 0.4293 | 0.4828 |
| Original 150: intended probe only | 0.7994 | 0.7945 |
| Original 150: cascade | 0.7839 | 0.7039 |
| Expansion 160: AED only | 0.1622 | 0.2034 |
| Expansion 160: intended probe only | 0.7216 | 0.7071 |
| Expansion 160: cascade | 0.7272 | 0.6927 |

The all-prediction denominator includes output labels with no matching target;
it exposes open-set false positives that a target-only macro average can hide.
Compared with mandatory closed-set probe emission, abstention changes OOD FPR
from 1.0 to 0.0682 on the original slice and 0.0227 on the expansion. Across all
310 clips, the probes emit on 20/440 OOD opportunities (0.0455): VocalSound
5/230 and FSD50K 15/210.

The affect ablation is unambiguous on these acted sentences. On the original
70 CREMA-D clips, emotion2vec+ macro-F1 is 0.9208 versus 0.0833 for the
transcript lexicon. On the ANG/FEA/DIS expansion, emotion2vec+ is 0.8679 while
the lexicon predicts neutral for all 60 clips and scores 0.0. This supports an
acoustic affect component; it does not establish internal emotional state.

## Confidence calibration

`scripts/calibrate.py` fits one scalar temperature per component by minimizing
categorical negative log likelihood on validation only. Probe validation joins
the original ID validation partition with its cross-domain `none` validation
examples. Affect uses 120 balanced, actor-disjoint CREMA-D clips from actors
1051–1060 across ANG, DIS, FEA, HAP, NEU, and SAD. No 310-clip inspection row is
used for fitting or selection.

| Component (inspection test) | Clips | Temperature | ECE before | ECE after | Brier before | Brier after |
|---|---:|---:|---:|---:|---:|---:|
| emotion2vec+ affect | 130 labelled / 310 total | 2.7058 | 0.0841 | 0.0679 | 0.1855 | 0.1506 |
| FSD50K class + `none` | 310 | 6.1112 | 0.1392 | 0.0670 | 0.2816 | 0.2474 |
| VocalSound class + `none` | 310 | 8.7453 | 0.0615 | 0.0244 | 0.1249 | 0.1031 |

These are explicitly **test** numbers from the untouched combined inspection.
The fit objective is NLL, not ECE or Brier. On affect validation, ECE improves
from 0.0953 to 0.0655 while Brier changes from 0.1987 to 0.2003; that small
validation Brier regression is retained rather than hidden. Both affect
metrics improve on test. Full validation/test values and partition labels are
in `research/calibration-results.json`.

Calibrated affect probabilities and top-label confidence now populate Attune
JSON. Emitted probe event/style confidence uses the calibrated class
probability. The PR #18 abstention boundary remains tied to its original
validation-selected uncalibrated class-minus-`none` margin, so temperature
scaling does not silently alter coverage or the reported OOD operating point.

## OOD and runtime summary

The calibrated rerun reproduces the PR #18 categorical and abstention results:
temperature scaling preserves argmax and the existing abstention decision.
Cross-source residual OOD errors are asymmetric:

| Probe | OOD source | False-positive clips |
|---|---|---:|
| VocalSound | CREMA-D | 0/130 |
| VocalSound | FSD50K | 5/100 |
| FSD50K | CREMA-D | 13/130 |
| FSD50K | VocalSound | 2/80 |

On this CPU rerun the full 310-clip cascade processes 1,488.748 seconds of audio
in 102.005 seconds, RTF 0.0685. By source, RTF is 0.0886 for CREMA-D, 0.0630
for VocalSound, and 0.0625 for FSD50K. This is offline batch timing on one VM,
not a streaming latency claim.

Separate ASR runs provide context. Ghanaian-English broadcast audio has
Whisper/SenseVoice WER 0.2562/0.2365 and RTF 0.0918/0.0193. British Common Voice
has WER 0.0997/0.1149 and RTF 0.1625/0.0305. Ghana is NC research-only and has
no speaker IDs; neither slice supports nationality or population claims.

## Timing and localization status

The 310-clip cascade still emits no word alignment and keeps all existing
event/style spans at 0..duration. VocalSound and FSD50K are clip-labelled, so
those spans are scope markers, not localization. The contract audit confirms
all 310 outputs remain schema-valid, deterministic in XML, structured-channel
separated, and utterance-timestamp-only.

Licence review identified DCASE 2016 Task 2 as a bounded exception suitable for
an independent timing diagnostic: original synthetic office mixtures, strong
onset/offset annotations, overlapping events, and recorded CC BY terms. The
committed 100-clip public-test manifest covers the Attune-overlapping laugh,
cough, and throat-clear labels. It does not retrofit timestamps onto
VocalSound/FSD50K or claim natural speech coverage.

## Gold review, errors, ethics, and non-goals

`attune.data.gold_review.GoldReviewRecord` defines a strict append-only review
ledger for transcript accept/reject, affect accept/reject/ambiguous, and event
or style accept/reject/retime/add decisions. Retimed/added spans require bounded
milliseconds. `docs/gold-review-protocol.md` specifies independent review and
adjudication. The committed six-row dry-run uses generated tones/noise and sets
`dry_run_fixture=true`; no real row is marked gold.

Machine-readable affect confusion, event/style TP/FP/FN, probe OOD-by-source,
and runtime tables are in `research/error-analysis/cascade-310.json`.
Interpretive notes are in `research/error-analysis/phase1-notes.md`. Notable
boundaries include CREMA `DIS` → `other`, 14/20 correct FEA clips, no
`crying_speech` target coverage, and residual FSD50K-head emissions on CREMA-D.

Attune estimates audible expression, not diagnosis, intent, truthfulness,
protected traits, or internal emotional state. It is not validated for
high-stakes decisions. Acted and synthetic labels, speaker gaps, NC restrictions,
and domain shifts remain visible in every report.

## Final Phase 1 gate decision

The engineering checklist is covered: ASR, behaviour labels, calibration,
abstention, OOD slices, runtime, error analysis, timing status, licence records,
and a usable human-review protocol all have committed evidence. The scientific
gate is nevertheless **CLOSED** because the principal 310 labels are not
human-adjudicated gold, most behaviour spans are not localized, subgroup and
in-the-wild evidence are narrow, and `crying_speech` is unmeasured. Completing
the checklist is not permission to rewrite those limitations as success.
