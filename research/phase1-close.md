# Phase 1 close

Phase 1 established a runnable, offline-first baseline and exposed the limits
that a proper training run must address. It did not create gold labels or real
event localization, and the research gate remains closed.

## What shipped

- SenseVoice-Small transcript and acoustic-event tags in a schema-valid cascade;
- emotion2vec+ acoustic affect, kept separate from transcript sentiment;
- frozen-SenseVoice linear VocalSound and FSD50K event/style probes;
- validation-selected `none`-logit abstention for both probes;
- deterministic Attune JSON and XML projection; and
- WER slices on CREMA-D acted US English, British Common Voice, and
  Ghanaian-English broadcast speech (the Ghana slice is NC research-only).

The combined PR #18 inspection used 310 weak source/acted labels, not reviewed
gold:

| Slice | WER | Affect macro-F1 | Target event/style macro-F1 | OOD FPR | All-prediction micro-F1 |
|---|---:|---:|---:|---:|---:|
| Original 150 | 0.0703 | 0.9208 | 0.7839 | 0.0682 | 0.7039 |
| Licence-clean expansion 160 | 0.1900 | 0.8679 | 0.7272 | 0.0227 | 0.6927 |

The separate 100-clip Ghanaian-English slice produced WER 0.2562 for
Whisper-Small and 0.2365 for SenseVoice-Small (roughly 0.24 overall context).
On the 100-speaker British Common Voice slice, Whisper-Small produced 0.0997
WER and SenseVoice-Small 0.1149. These slices differ in domain and are not
head-to-head population claims.

## Remaining gaps

- No row has been human-reviewed into gold.
- Event/style spans still cover the whole utterance; clip labels are not
  timestamps.
- Ghanaian-English WER remains materially worse than the acted-US and British
  slices, and the NC corpus cannot enter a commercial training set.
- There is no trained `crying_speech` style target.
- VocalSound and selected FSD50K clips are bounded diagnostics, not
  speech-embedded, in-the-wild coverage.
- Calibration and broader open-world/OOD evaluation remain incomplete.

## Decision

Gate: **closed for gold claims**. This Phase 1 closure does not turn weak labels
into gold, permit restricted datasets, or authorize encoder fine-tuning. The
SenseVoice-Small encoder remains frozen. SenseVoice-Small remains attributed to
FunASR/FunAudioLLM under its model agreement; emotion2vec+ to emotion2vec and
FunASR/FunAudioLLM; Whisper to OpenAI; VocalSound to Gong, Yu, and Glass;
FSD50K to Fonseca et al.; and CREMA-D to Cao et al.
