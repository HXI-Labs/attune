# Gold-source licence memo

Operational research notes, not legal advice. Audio and weights stay out of
git. Nothing here authorizes a download except material already on disk.

The scientific gold gate is **closed**. Phase 1 closed honestly: current event
and style targets are weak source or acted labels, or 100 ms human activity
that Attune has not independently reviewed.

## Recommendation

1. Review the 48-event STARSS23 first-60s pack already on disk
   (`data/manifests/starss23-gold-review-pack.jsonl`) under
   `docs/gold-review-starss23.md`.
2. Fetch ICSI only after Jerry green-lights a new download, and only after a
   separate laugh-timestamp licence and format review.
3. Do not fetch MULAI, Switchboard/LDC, or IB audiovisual corpora in this
   workstream.

## STARSS23 — best current natural laugh timing on disk

- Record: [Zenodo 7880637](https://zenodo.org/records/7880637), MIT licence.
- Provenance already in-tree: `data/provenance/starss23.yaml`.
- Human 100 ms activity plus optically tracked spatial position. Class 4 maps
  to Attune `laugh`. Language unverified. Natural overlapping speech.
- Eligible for Attune gold **only** after two independent Attune review passes
  and adjudication. The frozen frame-MLP collar F1 of 0.1395 is not a gold
  substitute.
- **Do not download again.** Use the existing first-60s mean-of-4 16 kHz cache.

## VocalSound — clip-level acted probe, not timestamp gold

- CC BY-SA 4.0. Official project plus transport mirror documented in
  `data/provenance/vocalsound.yaml`.
- Clip-level acted/crowdsourced laughs with no millisecond onsets.
- Already used as the frozen event probe. Not a timestamp gold source.

## DCASE 2016 Task 2 — synthetic isolated events

- CC BY 3.0 (train/dev) and CC BY 4.0 (public test with annotations). See
  `data/provenance/dcase2016_task2.yaml`.
- Strong onset/offset labels on synthetic sequences. The wired DCASE frame-head
  is a timing diagnostic, **not** natural gold.

## ICSI Meeting Corpus — next candidate, do not fetch in this PR

- Access: [Edinburgh AMI ICSI page](https://groups.inf.ed.ac.uk/ami/icsi/) and
  [download chooser](https://groups.inf.ed.ac.uk/ami/icsi/download/).
- Licence page: [CC BY 4.0](https://groups.inf.ed.ac.uk/ami/icsi/license.shtml).
- About 70 hours of English meetings. Laugh appears in NXT / original
  transcripts as `VocalSound` laugh and as `Comment` laughed-speech.
- Strong next natural-speech candidate **after** the 48-event STARSS23 pack
  exists as a review unit.
- Do **not** download 70 h here. Laugh timestamps still need a separate
  licence-plus-format review before fetch (NXT vs MRT, close-talk vs mix,
  discrete laugh vs laughed speech).

## MULAI — not fetchable now

- Jansen, Truong, Nazareth, Heylen, LREC 2020,
  [Introducing MULAI](https://aclanthology.org/2020.lrec-1.534.pdf).
- Individual licence; contact the authors (University of Twente Human Media
  Interaction). Questionnaire, video, audio, and physiology are not public
  drop-in downloads.

## Interaction Behavior Dataset — annotations, not a drop-in audio corpus

- Annotations: Zenodo [14759144](https://doi.org/10.5281/zenodo.14759144),
  repo [numediart/ib_dataset](https://github.com/numediart/ib_dataset).
- Audiovisual media come from CCDb, IFADV, and NDC-ME with mixed access
  (NDC-ME is contact-the-authors). This is not a drop-in audio corpus for
  Attune gold.

## Switchboard / LDC

Assume **not licence-clean** unless a later review proves otherwise. Do not
fetch.

## What this memo does not do

No encoder unfreeze, no Phase 3, no new MLP/GRU/Conv, no decoder grid, no
tiling, no ICSI/MULAI/Switchboard download, and no gold claim.
