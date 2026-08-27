# Phase 2 frozen-encoder probe package

Phase 2 is complete as a small-probe research package. SenseVoice-Small remains
fully frozen: all 233,999,167 encoder parameters have `requires_grad=False`,
frontend dither is `0.0`, extraction calls the frontend and encoder directly,
four query frames are excluded, and temporal statistics form a 5,120-value
embedding. No LoRA, encoder fine-tuning, joint training, streaming, ONNX, or
quantization was performed. Probe heads, embeddings, weights, and audio remain
local and gitignored.

The evidence is weak-source/acted inspection, not reviewed Attune gold. The
scientific gold gate remains **closed**.

## Frozen probes and cascade

| Evidence | Inspection result | Interpretation |
|---|---:|---|
| VocalSound five-way linear probe | macro-F1 0.8598 | Frozen features expose laugh, sigh, cough, throat-clear, and sneeze information beyond released AED tags. |
| FSD50K four-way linear probe | macro-F1 0.7410 | Frozen features separate source-labelled shout, whisper, sob, and scream; standalone clips do not establish speech-embedded style. |
| Cascade, original 150 | WER 0.0703; affect F1 0.9208; target event F1 0.7839; OOD FPR 0.0682 | AED plus validation-abstaining probes adds useful utterance-level coverage. |
| Cascade, expansion 160 | WER 0.1900; affect F1 0.8679; target event F1 0.7272; OOD FPR 0.0227 | The result persists on held-out ANG/FEA/DIS and FSD50K material, with domain limitations. |

The VocalSound and FSD50K heads have 25,605 and 20,484 trainable parameters.
Each uses genuine cross-domain negatives and a validation-selected `none`-logit
abstention boundary. Calibration temperatures are also validation-only.
SenseVoice AED is merged first, followed by VocalSound and FSD50K, as a
structured set union. `scream` remains an event rather than `shouting`; `sob`
remains an event rather than `crying_speech`; CREMA-D intensity creates no
style; and `DIS` maps to Attune `other`.

Sources: `research/sensevoice-frozen-probe-metrics.json`,
`research/fsd50k-frozen-probe-metrics.json`, and
`research/attune-cascade-inspection-results.json`.

## Operational affect abstention

The previous emotion2vec+ adapter always emitted a category. Phase 2 fits a
maximum-probability rule after validation temperature scaling. The threshold
0.8337396463 was selected only on 120 balanced CREMA-D clips from held-out
actors 1051–1060. Selection maximized retained-only macro-F1 subject to at least
80% validation coverage; ties preferred higher coverage. None of the 310
cascade inspection IDs, including the 130 labelled CREMA-D test clips, entered
selection.

The hypothesis that a confidence threshold would improve the quality of emitted
labels was retained, with its coverage cost reported:

| 130-clip acted CREMA inspection | Always emit | Validation-fitted abstention |
|---|---:|---:|
| Coverage | 1.0000 (130/130) | 0.7615 (99/130) |
| Abstained | 0 | 31 |
| Macro-F1 on emitted clips | 0.8829 | 0.9819 |
| ECE on emitted clips | 0.0679 | 0.0295 |
| Brier on emitted clips | 0.1506 | 0.0413 |
| Full-population macro-F1, abstentions counted as misses | 0.8829 | 0.8372 |

Retained-only metrics condition on the 76.2% coverage and must not be read as
full-population accuracy. Abstention improves selective quality but lowers
full-population macro-F1 when withheld labels count as misses. Before
temperature scaling, always-emit ECE/Brier were 0.0841/0.1855; scaling changed
them to 0.0679/0.1506 before abstention. Validation itself retained 98/120
clips (81.7%), with retained macro-F1 0.9384 versus 0.8987 always-emitted.

At runtime, a score below the threshold produces `affect.abstain=true`,
`affect.top_label=null`, and the complete calibrated category distribution.
The structured output remains schema-valid; no affect label is inserted into
transcript text. The fit and test rerun used acted labels only and created no
gold record.

## Local WAV inference

`scripts/infer.py` accepts one or more local WAV files and emits authoritative
Attune JSON plus optional deterministic XML. Real inference requires explicit
local SenseVoice-Small, emotion2vec+, VocalSound-head, and FSD50K-head paths,
the Phase 2 calibration bundle, and
`ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1`. A missing item is a hard error; the CLI
does not download weights or silently degrade to a partial cascade.

The weight-free `--fixture-mode` path verifies WAV reading, schema output,
abstention invariants, and serialization while naming itself as a placeholder.
It is not a model result. Exact real and fixture commands are in
`docs/baseline-runners.md`.

The current ASR adapter has utterance text but no reviewed alignment, so the
authoritative `transcript.words` list is empty. Events and styles use
provisional null-anchored `0..duration` bounds solely to encode utterance scope;
those bounds are not presented as localization.

## Timing boundary

The DCASE 2016 temporal MLP remains a negative result: segment F1 0.1943 versus
0.3183 for the whole-clip comparator, with 0 collar event F1 for both. It is not
wired into cascade timestamps, and this PR does not rerun it.

No small pre-cached, hash-verified natural-audio timing slice was present.
STARSS23 is therefore only the next Phase-2-adjacent candidate: it is the MIT
natural-spatial-audio dataset with 100 ms labels, English cannot be selected
from its metadata, and licence/privacy review is required for natural
recordings. No STARSS23 or DCASE download was started for this package.

## What Phase 2 established

- Small linear heads over one frozen sub-300M encoder provide useful
  utterance-level event/style evidence on the bounded acted/source-labelled
  inspections.
- Validation-selected probe abstention materially lowers cross-domain false
  emissions while preserving useful target F1.
- Validation temperature scaling improves test ECE/Brier, and affect can now
  withhold low-confidence labels through the authoritative schema.
- A complete offline local-cascade CLI can preserve trusted structured
  channels without a product UI or model downloads.
- The attempted frozen temporal head did not justify finer timestamps.

## What still requires Phase 3 evidence

- independently reviewed and adjudicated real gold labels;
- proof that joint or encoder optimization is necessary and ethically
  supportable, rather than merely larger;
- natural-audio timing that beats honest utterance scope;
- speech-embedded styles, broader open-world OOD and recording-quality slices,
  subgroup checks, and improved Ghanaian-English evidence; and
- a new gate decision before any encoder unfreezing.

Phase 3 is not started by this report. `scripts/train_joint.py` and
`scripts/export_onnx.py` remain stubs.

## Licences and use boundary

SenseVoice-Small and emotion2vec+ retain their model names and attribution to
FunASR/FunAudioLLM under the FunASR Model Open Source License Agreement v1.1.
Whisper remains attributed to OpenAI under MIT. VocalSound, FSD50K, and
CREMA-D terms and hashes remain in `data/provenance/`; Ghana material remains
NC research-only. Attune describes perceived acoustics, not internal state,
diagnosis, deception, risk, or suitability for high-stakes automated
decisions. It is not a consumer voice assistant.
