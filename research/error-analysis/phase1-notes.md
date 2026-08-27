# Phase 1 error analysis

These notes interpret bounded weak/acted-label inspections. They are not a
review of individual speakers and do not establish gold performance.

## ASR

- SenseVoice WER is lowest on acted CREMA-D US English (0.0703 in the original
  cascade slice), 0.1149 on the 100-speaker British Common Voice slice, and
  0.2365 on Ghanaian-English broadcast speech.
- Whisper is slightly better on the British slice (0.0997) and worse on the
  Ghana slice (0.2562). Domain, named entities, recording conditions, and accent
  all change together, so these are error slices rather than accent rankings.
- The Ghana corpus is CC BY-NC and lacks speaker IDs. Its errors cannot justify
  commercial training use or speaker-generalization claims.

## Affect

- emotion2vec+ materially outperforms the transcript lexicon ablation on acted
  acoustic delivery. The lexicon cannot distinguish same-text/different-delivery
  cases and can follow explicit emotional words when acoustics conflict.
- CREMA-D `DIS` maps to Attune `other`, never `distress`. That ontology boundary
  should be read before treating `other` confusions as model failures.
- Confidence is evaluated against acted source labels, including ANG, FEA, DIS,
  HAP, NEU, and SAD. Temperature scaling changes confidence, not the top-label
  ordering; it does not make the labels subjective-consensus gold.

## Events, styles, and OOD

- SenseVoice AED misses most of the target distinctions exposed by the frozen
  probes. The probe-only and AED-only ablations therefore measure different
  coverage rather than interchangeable classifiers.
- The PR #18 `none` logits sharply reduce cross-domain emissions, but residual
  OOD false positives remain source-dependent. See `cascade-310.json` for
  per-source rates and per-label TP/FP/FN counts.
- FSD50K `Screaming` remains the event `scream`, not the style `shouting`;
  `Crying_and_sobbing` remains `sob`, not `crying_speech`. No Phase 1 corpus
  establishes `crying_speech` coverage.
- VocalSound and FSD50K targets are clip-level. Their 0..duration spans remain
  explicit utterance scope and must not be scored as localization.

## Timing

- The DCASE diagnostic is synthetic office audio with strong onset/offset
  labels for laugh, cough, and throat-clear. It can test whether frozen
  SenseVoice temporal bins carry boundary information, but it is not
  speech-embedded in-the-wild evidence.
- Its whole-clip comparator is intentionally generous in label identity and
  intentionally wrong in time: it uses the known clip label set and spans each
  label across the full ten seconds. Segment and collar metrics determine
  whether the temporal head adds actual localization.

## User-facing consequences

Keep event/style status provisional, retain the perceptual-affect warning, and
surface abstention instead of forcing a label. Do not infer diagnosis, intent,
truthfulness, protected traits, or internal emotional state. High-stakes use is
outside the Phase 1 evidence.
