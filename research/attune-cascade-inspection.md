# Attune cascade combined inspection

The authoritative machine-readable report is
`research/attune-cascade-inspection-results.json`. It records one offline run
over all 310 clips: the original 150-clip inspection set and the 160-clip
licence-clean expansion. Targets remain weak source/acted labels, not reviewed
Attune gold.

## Cascade inspected

- transcript and AED: SenseVoiceSmall;
- affect: emotion2vec+ acoustic SER;
- events/styles: SenseVoice AED union the frozen-SenseVoice VocalSound linear
  probe union the frozen-SenseVoice FSD50K linear probe; and
- output: schema-valid JSON first, with deterministic XML projection.

SenseVoice AED is merged first. The probes fill labels not already present;
later duplicates are suppressed. `scream` is a discrete event and never maps to
`shouting`. `sob` never implies `crying_speech`. All annotations are provisional
whole-utterance spans. No word or frame localization is claimed.

## Results

| Slice | ASR WER | emotion2vec+ accuracy / macro-F1 | lexicon accuracy / macro-F1 |
|---|---:|---:|---:|
| Original 150 (70 transcript/affect clips) | 0.0703 | 0.9286 / 0.9208 | 0.1429 / 0.0833 |
| Expansion 160 (60 transcript/affect clips) | 0.1900 | 0.8500 / 0.8679 | 0.0000 / 0.0000 |

| Event/style slice | Cascade target macro-F1 | AED-only | Intended probe-only |
|---|---:|---:|---:|
| Original VocalSound 80 | 0.7912 | 0.4498 | 0.8370 |
| Expansion FSD50K 100 | 0.7465 | 0.1429 | 0.7410 |

“Intended probe-only” means the VocalSound probe on VocalSound and the FSD50K
probe on FSD50K. The JSON also reports the literal union of both closed-set
probes. Because each head must choose one class even out of domain, the cascade
micro-F1 over every emitted label is only 0.5075 on VocalSound and 0.4934 on
FSD50K. This is the key reason the gate remains closed: no OOD or abstention
threshold has been evaluated.

The fresh gitignored VocalSound head was trained on the exact recorded
514/132/80 speaker-disjoint split and measured test macro-F1 0.8370. This is
lower than the previously committed 0.8598 run; the combined report uses the
fresh head rather than substituting the historical score. The fresh FSD50K head
reproduced test macro-F1 0.7410 on the clip-disjoint 256/64/100 split.

## Contract and training audit

All 310 outputs passed schema validation, deterministic XML rendering,
utterance-only timestamp checks, and transcript/metadata channel-separation
checks. The SenseVoice encoder reported zero trainable parameters, direct
frontend/encoder extraction, frontend dither `0.0`, and 5,120-dimensional
pooling. Only 25,605 VocalSound-head parameters and 20,484 FSD50K-head
parameters were trainable. Audio, model weights, embeddings, and `.pt`
checkpoints remain gitignored.

## Decision

Gate: **closed**. The cascade is executable and its coverage gains are visible,
but the cross-domain false positives from mandatory closed-set probe decisions
must be addressed with evaluated abstention/OOD behavior before promotion.
