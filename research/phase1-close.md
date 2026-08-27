# Phase 1 close

Phase 1 now covers the baseline checklist: ASR, behaviour labels, timing status,
confidence calibration, abstention, OOD slices, runtime, error analysis,
licences, and a usable gold-review protocol. It established an offline-first
reference implementation and also produced negative evidence. It did not turn
source labels into gold, and the scientific gate remains closed.

## System and methods

The combined cascade uses:

- SenseVoice-Small for utterance transcript and released acoustic-event tags;
- emotion2vec+ for acoustic affect, never transcript sentiment;
- a five-way VocalSound linear probe over frozen SenseVoice embeddings;
- a four-way FSD50K linear probe over the same frozen encoder;
- one fitted `none` row per probe with validation-selected abstention margins;
- validation-fitted scalar temperatures for affect and probe class/`none`
  probabilities; and
- schema-valid JSON as authority, with deterministic XML derived afterward.

SenseVoice extraction calls the official frontend and encoder directly, fixes
dither to `0.0`, excludes query frames, and pools eight temporal means plus
mean/std. All 233,999,167 encoder parameters remain frozen. No model weights,
embeddings, audio, or `.pt` heads are committed.

## Evaluation material

| Material | Use | Split/label status |
|---|---|---|
| CREMA-D | ASR and affect, including ANG/FEA/DIS | actor-disjoint bounded slices; acted labels |
| VocalSound | laugh/sigh/cough/throat-clear/sneeze | speaker-disjoint probe and 80-clip test; clip labels |
| selected FSD50K | shout/whisper/sob/scream | clip-disjoint CC0/CC BY selection; clip labels |
| Common Voice 17 English | British WER | 100 distinct client IDs; CC0 |
| Ghana English ASR | Ghanaian-English WER | 100 clips; CC BY-NC; no speaker IDs |
| DCASE 2016 Task 2 | timing diagnostic | synthetic strong onset/offset; CC BY; source-disjoint test |

The combined 310 inspection comprises the original 150 CREMA-D/VocalSound
clips and the 160-clip CREMA-D/FSD50K expansion. Every target remains a weak
source or acted label, not reviewed Attune gold.

## Combined results

| Slice | WER | Affect macro-F1 | Target event/style macro-F1 | OOD FPR | All-prediction micro-F1 |
|---|---:|---:|---:|---:|---:|
| Original 150 | 0.0703 | 0.9208 | 0.7839 | 0.0682 | 0.7039 |
| Licence-clean expansion 160 | 0.1900 | 0.8679 | 0.7272 | 0.0227 | 0.6927 |

The calibrated rerun reproduces these label decisions because temperature
scaling preserves argmax and abstention retains its original validation-tuned
margin. Across both slices, the probes emit on 20/440 OOD opportunities.

The event/style ablations show why the cascade is composite. Original-slice
target macro-F1 is 0.4293 for AED only, 0.7994 for intended probe only, and
0.7839 for the union. Expansion values are 0.1622, 0.7216, and 0.7272. The
union gains coverage but also retains AED false positives; all-prediction
micro-F1 keeps that cost visible.

The affect ablation separates acoustics from words. emotion2vec+ scores 0.9208
macro-F1 on the original acted CREMA-D affect slice versus 0.0833 for the
transcript lexicon. On ANG/FEA/DIS, emotion2vec+ scores 0.8679 while the
lexicon predicts neutral throughout and scores 0.0. `DIS` maps to Attune
`other`, not `distress`.

## ASR and runtime slices

The separate Ghanaian-English broadcast slice produces WER 0.2562 for
Whisper-Small and 0.2365 for SenseVoice-Small, with RTF 0.0918 and 0.0193.
The British Common Voice slice produces WER 0.0997 and 0.1149, with RTF 0.1625
and 0.0305. The domains differ and Ghana lacks speaker IDs, so the table is not
an accent or population ranking. Ghana audio is NC research-only and cannot be
repurposed for commercial training.

The final 310-clip cascade processes 1,488.748 seconds of audio in 102.050
seconds on CPU (RTF 0.0685). This is offline batch throughput, not streaming
latency.

## Calibration

`scripts/calibrate.py` minimizes categorical NLL using validation only. Affect
uses 120 balanced CREMA-D clips from held-out actors 1051–1060. Probe
calibration combines each ID validation partition with cross-domain
validation examples labelled `none`. The 310 inspection is test only.

| Component (test) | Clips | Temperature | ECE before → after | Brier before → after |
|---|---:|---:|---:|---:|
| emotion2vec+ affect | 130 labelled / 310 total | 2.7058 | 0.0841 → 0.0679 | 0.1855 → 0.1506 |
| FSD50K class + `none` | 310 | 6.1112 | 0.1392 → 0.0670 | 0.2816 → 0.2474 |
| VocalSound class + `none` | 310 | 8.7453 | 0.0615 → 0.0244 | 0.1249 → 0.1031 |

On affect validation, ECE improves but Brier slightly regresses
(0.1987 → 0.2003); the NLL-selected fit is retained and the regression is
reported. Attune JSON now carries calibrated affect distributions/top-label
confidence and calibrated confidence for emitted probe annotations.

## Timing result

The 310 cascade has no word alignment and keeps VocalSound/FSD50K behaviour
spans at utterance scope. Their clip tags are not localization.

After licence review, a separate DCASE diagnostic used 168 development clips
for fitting, 48 development clips for threshold selection, and a bounded
100-clip source-disjoint public test. A 66,051-parameter temporal MLP consumed
eight frozen SenseVoice bins. It scores 0.1943 one-second segment F1 and 0.0
200 ms collar event F1. The whole-clip oracle-tag comparator scores 0.3183 and
0.0. The temporal head therefore does **not** beat whole-clip and is not wired
into cascade timestamps. This negative result prevents fake localization.

## Review, errors, and non-goals

`GoldReviewRecord` and `docs/gold-review-protocol.md` define append-only
accept/reject/retime/add decisions with hash and duration checks. A six-row
fixture dry-run proves the contract on generated tones/noise only; no real row
is marked gold.

Machine-readable affect confusion, event/style TP/FP/FN, OOD-by-source, and
runtime tables are under `research/error-analysis/`. Residual failures include
13/130 FSD50K-head emissions on CREMA-D, five VocalSound-head emissions on
FSD50K, 14/20 correct fear clips, no `crying_speech` target, and coarse timing.

Attune estimates audible expression. It does not infer diagnosis, truthfulness,
intent, protected traits, or internal emotional state, and Phase 1 does not
validate high-stakes use.

## Remaining gaps and decision

- No real row has independent review and adjudication into gold.
- Existing cascade event/style outputs remain utterance-scoped.
- The DCASE timing set is synthetic and covers only laugh, cough, throat-clear.
- Ghanaian-English WER is materially worse and available evidence is NC.
- `crying_speech`, broader in-the-wild behaviour, recording-quality slices, and
  open-world OOD remain incomplete.

Gate: **CLOSED for scientific gold claims**. Phase 1 checklist completion is
documented evidence, not permission to erase these limitations. The
SenseVoice-Small encoder remains frozen. SenseVoice-Small is attributed to
FunASR/FunAudioLLM under its model agreement; emotion2vec+ to emotion2vec and
FunASR/FunAudioLLM; Whisper to OpenAI; VocalSound to Gong, Yu, and Glass;
FSD50K to Fonseca et al.; CREMA-D to Cao et al.; and the DCASE timing data to
Grégoire Lafay/IRCCYN and the DCASE 2016 challenge.
