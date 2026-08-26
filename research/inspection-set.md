# Phase 1 inspection set

The first manifest selects **150 acted clips** for qualitative inspection and
slice checks. It is not a sealed gold baseline and is not training data.
`data/manifests/inspection-set.jsonl` contains 116 `inspect` clips and 34
`held_out_speakers` clips. All 26 namespaced speaker IDs occur in exactly one
partition.

The 2026-08-26 licence review remains operational research guidance, not legal
advice or lawyer sign-off. Audio stays in a gitignored local cache. The
repository contains only manifests, selected-file provenance hashes, fetch
instructions, and attribution.

## What this subset covers

### VocalSound

- 80 16 kHz clips from 16 speakers: eight filename-marked female and eight
  filename-marked male speakers.
- Balanced source events: 16 each of laughter, sigh, cough, throat clearing,
  and sneeze.
- Source-to-Attune weak-label mappings:
  `laughter → laugh`, `sigh → sigh`, `cough → cough`,
  `throatclearing → throat_clear`, and `sneeze → sneeze`.
- Sniffs are deliberately excluded; none are mapped to `breath`.
- These are standalone crowdsourced acted events, not events embedded in
  natural speech.

VocalSound is by Yuan Gong, Jin Yu, and James Glass, ICASSP 2022,
[doi:10.1109/ICASSP43922.2022.9746828](https://doi.org/10.1109/ICASSP43922.2022.9746828),
and is licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The versioned
Zenodo WebDataset record is used only as transport because the official
Dropbox URL no longer yields an archive to the fetch client; the official
VocalSound identity, attribution, and ShareAlike terms still govern.

### CREMA-D

- 70 AudioWAV clips from 10 actors: five female and five male according to the
  official demographics table.
- For every actor, three sentence groups provide same-speaker delivery
  contrasts:
  - neutral text, “It's eleven o'clock”: neutral/unspecified, happy/high, and
    sad/high;
  - mildly positive text, “I would like a new alarm clock”: happy and sad
    deliveries;
  - mildly negative text, “The surface is slick”: happy and sad deliveries.
- Source delivery labels map weakly to Attune affect categories:
  `NEU → neutral`, `HAP → joy`, and `SAD → distress`. CREMA-D intensity is
  retained as source metadata. It is not mapped to `shouting`, vocal effort, or
  another style without human listening.
- Every clip is explicitly marked as acted.

CREMA-D is by Cao et al., IEEE Transactions on Affective Computing 5(4), 2014,
[doi:10.1109/TAFFC.2014.2336244](https://doi.org/10.1109/TAFFC.2014.2336244).
Its database is licensed under
[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) and individual
contents under the Database Contents License. Attribution and database
ShareAlike obligations must be preserved.

## What this subset still lacks

This first set does **not** cover:

- Ghanaian English or verified British English;
- crying speech or sob events;
- verified shouting or whispering;
- true quiet high-effort speech or a defensible loud-neutral contrast;
- naturalistic, spontaneous affect;
- inline vocal events within speech;
- source-labelled fear, anger, surprise, or ambiguous delivery; or
- independently reviewed recording-level, clipping, transcript-polarity, and
  source-label correctness.

RAVDESS, Ghana English ASR, SAVEE, DEED, EmoV-DB, and MSP-Podcast were not
downloaded or included. Filling the listed gaps requires a separate
licence/consent decision and a new manifest revision.

## Review boundary

All event, affect, intensity, and acted fields originate from source labels or
source documentation. They are weak labels, not claims about internal emotion
and not Attune gold. Human review is still required before any row is treated
as gold, including listening for label correctness, delivery contrast,
recording quality, clipping, and suitability for the intended slice.

No fine-tuning is authorised by this manifest. The baseline-report stage gate
and the ban on committing model weights remain in force.
