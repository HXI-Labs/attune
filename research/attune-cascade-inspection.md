# Attune cascade combined inspection

The authoritative machine-readable report is
`research/attune-cascade-inspection-results.json`. It records one offline run
over all 310 clips: the original 150-clip inspection set and the 160-clip
licence-clean expansion. Targets remain weak source/acted labels, not reviewed
Attune gold.

## Cascade inspected

- transcript and AED: SenseVoiceSmall;
- affect: emotion2vec+ acoustic SER;
- events/styles: SenseVoice AED union validation-selected abstaining
  frozen-SenseVoice VocalSound and FSD50K linear probes; and
- output: schema-valid JSON first, with deterministic XML projection.

SenseVoice AED is merged first. An abstaining probe contributes no annotation,
so AED-only output is retained; when every source is empty, events/styles remain
empty. Later duplicates are suppressed. `scream` is a discrete event and never
maps to `shouting`. `sob` never implies `crying_speech`. All annotations are
provisional whole-utterance spans. No word or frame localization is claimed.

## Results

| Slice / metric | PR #17 | Abstaining probes | Delta |
|---|---:|---:|---:|
| Original: ASR WER | 0.0703 | 0.0703 | -0.0000 |
| Original: emotion2vec+ macro-F1 | 0.9208 | 0.9208 | -0.0000 |
| Original: cascade target macro-F1 | 0.7912 | 0.7839 | -0.0073 |
| Original: AED-only target macro-F1 | 0.4498 | 0.4293 | -0.0205 |
| Original: intended probe-only target macro-F1 | 0.8370 | 0.7994 | -0.0376 |
| Original: OOD false-positive rate | 1.0000 | 0.0682 | -0.9318 |
| Original: all-prediction micro-F1 | 0.5075 | 0.7039 | +0.1964 |
| Expansion: ASR WER | 0.1900 | 0.1900 | 0.0000 |
| Expansion: emotion2vec+ macro-F1 | 0.8679 | 0.8679 | +0.0000 |
| Expansion: cascade target macro-F1 | 0.7465 | 0.7272 | -0.0193 |
| Expansion: AED-only target macro-F1 | 0.1429 | 0.1622 | +0.0193 |
| Expansion: intended probe-only target macro-F1 | 0.7410 | 0.7216 | -0.0194 |
| Expansion: OOD false-positive rate | 1.0000 | 0.0227 | -0.9773 |
| Expansion: all-prediction micro-F1 | 0.4934 | 0.6927 | +0.1993 |

“Intended probe-only” remains the VocalSound probe on VocalSound and the
FSD50K probe on FSD50K. OOD false-positive rate is the fraction of
out-of-domain clip/probe opportunities with any emission; mandatory PR #17
closed-set decisions therefore have rate 1.0. Across all 310 clips, the
abstaining heads emit on 20/440 OOD opportunities (0.0455): VocalSound 5/230
(0.0217) and FSD50K 15/210 (0.0714). All-clips/all-label micro-F1 is 0.6751.

The OOD failure is materially reduced while target macro-F1 does not collapse:
the cascade changes by -0.0073 on VocalSound and -0.0193 on FSD50K. The larger
probe-only changes (-0.0376 and -0.0194) make the selective-coverage tradeoff
explicit rather than hiding abstentions.

SenseVoice ASR/AED was also rerun with frontend dither fixed at `0.0`; the small
AED-only deltas are therefore reported rather than assumed away. WER and
emotion2vec+ affect reproduce the PR #17 values at four decimals.

## Abstention selection

Both heads compare max-softmax, negative energy (`logsumexp(logits)`), and a
genuine-negative `none` logit. The original class rows are retained exactly;
only one scalar `none` row is fitted. Its score is maximum class probability
minus `none` probability, and emission requires score ≥ threshold. FSD50K and
VocalSound supply cross-domain negatives; 90 additional neutral CREMA speech
clips use actors 1031–1050 for training and 1051–1060 for validation. Inspection
actors are 1001–1030.

| Probe / validation-selected candidate | Threshold | ID macro-F1 | OOD FPR | All-prediction micro-F1 |
|---|---:|---:|---:|---:|
| VocalSound max-softmax | 0.999499 | 0.7922 | 0.8617 | 0.5938 |
| VocalSound energy | 26.392059 | 0.7615 | 0.7234 | 0.5941 |
| **VocalSound `none` margin (chosen)** | **0.997538** | **0.7884** | **0.0106** | **0.7866** |
| FSD50K max-softmax | 0.999999 | 0.7952 | 0.5494 | 0.4478 |
| FSD50K energy | 13.742188 | 0.8961 | 0.9074 | 0.4118 |
| **FSD50K `none` margin (chosen)** | **0.718345** | **0.9209** | **0.0309** | **0.8855** |

Small validation threshold sweeps (the validation-chosen row is marked `*`):

| Probe | `none` threshold | ID macro-F1 | OOD FPR | All-prediction micro-F1 |
|---|---:|---:|---:|---:|
| VocalSound | -1.000000 | 0.7748 | 1.0000 | 0.5698 |
| VocalSound | 0.997523 | 0.7859 | 0.0106 | 0.7833 |
| VocalSound | **0.997538*** | **0.7884** | **0.0106** | **0.7866** |
| VocalSound | 0.998477 | 0.7838 | 0.0106 | 0.7815 |
| FSD50K | -1.000000 | 0.9219 | 1.0000 | 0.4069 |
| FSD50K | 0.686580 | 0.9209 | 0.0370 | 0.8788 |
| FSD50K | **0.718345*** | **0.9209** | **0.0309** | **0.8855** |
| FSD50K | 0.766699 | 0.9117 | 0.0309 | 0.8769 |

## Contract and training audit

All 310 outputs passed schema validation, deterministic XML rendering,
utterance-only timestamp checks, and transcript/metadata channel-separation
checks. The SenseVoice encoder reported zero trainable parameters, direct
frontend/encoder extraction, frontend dither `0.0`, and 5,120-dimensional
pooling. The exact PR #17 VocalSound 514/132/80 speaker-disjoint split and
FSD50K 256/64/100 clip-disjoint split were reused. Selected checkpoints contain
30,726 VocalSound and 25,605 FSD50K linear parameters, but class rows come from
the original 25,605/20,484-parameter closed-set heads and only one 5,121-
parameter `none` row is subsequently fitted. The encoder is never updated.
Audio, model weights, embeddings, and `.pt` checkpoints remain gitignored.

Attribution remains in the authoritative JSON for FunASR/SenseVoiceSmall,
emotion2vec+, Whisper, VocalSound, FSD50K, and CREMA-D. CREMA `DIS` remains
Attune `other`; intensity creates no style.

## Decision

Gate: **closed**. The bounded evidence now supports reconsideration: OOD
false-positive rates fall sharply and all-prediction micro-F1 rises without an
in-domain target-F1 collapse. This report does not pass the gate. Targets remain
weak source/acted labels, selective coverage costs remain visible, and broader
open-world OOD behavior still requires review.
