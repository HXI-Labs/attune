# Phase 1 inspection-set manifests

`inspection-set.jsonl` selects 150 acted clips from the only approved Phase 1
sources:

- 80 VocalSound 16 kHz event clips (16 speakers; 16 examples each of `laugh`,
  `sigh`, `cough`, `throat_clear`, and `sneeze`);
- 70 CREMA-D AudioWAV clips (10 actors) arranged as same-speaker,
  same-sentence delivery contrasts.

There are 116 `inspect` clips and 34 `held_out_speakers` clips. The partitions
are speaker-disjoint both within each source and across the namespaced speaker
IDs. Source labels are weak labels for inspection, not reviewed Attune gold.

## Fetch and verify locally

Audio and transport archives are ignored by git. From the repository root:

```bash
# Report missing files without network access.
uv run python scripts/prepare_dataset.py

# Fetch only the files/transport shards referenced by the manifest, then verify.
uv run python scripts/prepare_dataset.py --download

# A single partition can be prepared independently.
uv run python scripts/prepare_dataset.py --download --split inspect
```

The default cache is `data/raw/inspection-set/`; override it with
`--cache-dir`. CREMA-D WAV files are fetched individually from the official
repository at a pinned revision. The official VocalSound Dropbox link currently
does not return the archive to non-browser clients, so the manifest uses the
versioned Zenodo record
[`14650192`](https://doi.org/10.5281/zenodo.14650192) as a transport mirror for
the original 16 kHz files. Only ten small validation transport shards are
needed, rather than the full VocalSound corpus. The mirror does not change the
source identity or licence: the official VocalSound CC BY-SA 4.0 terms govern.

Every extracted or downloaded clip must match its manifest SHA-256. Shared
transport archives are cached under `_archives/` and checked against the
record's MD5 transport checksum before extraction. `inspection-set.provenance.json`
records SHA-256 aggregates over only the selected files, plus pinned upstream
revisions; it does not claim a full-corpus hash.

## JSONL fields

Each row records the required clip, source, filename, namespaced speaker,
partition, measured duration, local-file SHA-256, licence, attribution, intended
Attune event/style/affect weak labels, acted status, and review notes. Additional
fields provide source metadata, a relative cache path, and a reproducible fetch
recipe. `null` hashes are not used because all 150 selected files were fetched
and hashed locally.

## Attribution and licences

VocalSound is by Yuan Gong, Jin Yu, and James Glass:
“VocalSound: A Dataset for Improving Human Vocal Sounds Recognition,” ICASSP
2022, [doi:10.1109/ICASSP43922.2022.9746828](https://doi.org/10.1109/ICASSP43922.2022.9746828).
It is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Preserve
attribution and apply the licence's ShareAlike requirements to covered
adaptations.

CREMA-D is by Cao et al.:
“CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset,” IEEE Transactions
on Affective Computing 5(4), 2014,
[doi:10.1109/TAFFC.2014.2336244](https://doi.org/10.1109/TAFFC.2014.2336244).
The database is offered under the
[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) and its individual
contents under the Database Contents License. Preserve attribution and the
ODbL's database ShareAlike obligations.

These notices are operational research guidance, not legal advice. Original
audio is not committed or redistributed by this repository.

## Ghanaian-English WER slice — NC research-only

`ghana-english-wer.jsonl` selects 100 clips (1,328.911 seconds) from the pinned
`ghananlpcommunity/ghana-english-asr-2700hrs` revision. The fetcher streams only
the selected rows and converts them to mono 16 kHz PCM16 WAV under the ignored
`data/raw/ghana-english-wer/` cache:

```bash
uv sync --extra dataset-tools
uv run python scripts/prepare_ghana_english.py --download
uv run python scripts/prepare_ghana_english.py
```

Every row includes the source and converted-audio SHA-256, transcript, duration,
audio contract, pinned revision, cache path, fetch script, attribution, and
explicit `research_only` and `commercial_redistribution_prohibited` flags. The
published corpus schema has no speaker identifier, so speaker-disjoint sampling
and speaker leakage checks are impossible; the manifest records this limitation
and does not claim speaker disjointness.

The Ghana NLP Community dataset is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This manifest,
the local subset, and all reported results are **NC research-only**. They must
never be used in a commercially redistributed training set and are not covered
by this repository's MIT code licence. The preparation script must not run in
CI, and raw audio must remain outside Git.
