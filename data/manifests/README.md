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
and does not claim speaker disjointness. Selection is the first 100
duration-valid rows from the pinned stream, so it may include repeated speakers
or adjacent broadcast segments and is not population-representative.

The Ghana NLP Community dataset is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This manifest,
the local subset, and all reported results are **NC research-only**. They must
never be used in a commercially redistributed training set and are not covered
by this repository's MIT code licence. The preparation script must not run in
CI, and raw audio must remain outside Git.

## British-English WER slice — Common Voice CC0

`common-voice-british-wer.jsonl` selects 100 clips (550.030 seconds) from
Mozilla Common Voice Corpus 17.0 English, transported through the pinned
`fixie-ai/common_voice_17_0` mirror revision
`34f78a43893414e7b6e271ba94c1d5e05f18b239`. The exact self-declared `accent`
field values present in the committed slice are **`England English`** (96
clips) and **`Scottish English`** (4 clips). The configured British filter also
accepts `Welsh English`, but no Welsh row occurred before the deterministic
100-speaker target was reached. Accent metadata is self-declared and is not
independent verification of nationality or residence.

The fetcher streams only `client_id`, path, sentence, and accent metadata,
stopping at source row 1,798. It then fetches only the selected audio assets;
it never downloads a full Common Voice archive or Parquet shard. Selection
keeps the first duration-valid row for each distinct `client_id`, yielding 100
clips from 100 speakers. Each asset is converted to mono 16 kHz PCM16 WAV under
gitignored `data/raw/common-voice-british-wer/`:

```bash
uv sync --extra dataset-tools
uv run python scripts/prepare_common_voice_british.py --download
uv run python scripts/prepare_common_voice_british.py
```

Every manifest row records its source row, source and converted hashes,
transcript, exact accent value, `client_id`, duration, audio contract, pinned
mirror revision, cache path, and fetch script. The source is Mozilla Common
Voice under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
Mozilla Common Voice attribution is retained voluntarily. Do not attempt to
determine speaker identities. No audio is committed.

## Licence-clean event and affect expansion

`licence-clean-inspection.jsonl` adds 160 individually fetched, converted, and
SHA-256-hashed clips:

- 25 each of FSD50K `Shout`, `Whispering`, `Crying_and_sobbing`, and
  `Screaming` (100 clips total); and
- 20 CREMA-D actors × `ANG`, `FEA`, and `DIS` (60 clips), using actors absent
  from the original 70-clip inspection set.

The FSD50K selector downloads only the official 6.7 MB metadata and 335 KB
ground-truth archives first. It rejects every clip whose own Freesound licence
is not CC0 or CC BY before constructing an individual audio URL. It never
downloads the full FSD50K audio archive. Uploader, clip licence, title, source
class, FSD50K curation attribution, and the pinned transport revision are
retained per row. CC BY-NC and Sampling+ clips are ineligible.

```bash
# Fetch only the 160 selected files, with finite retry/backoff, and convert.
uv run python scripts/prepare_licence_clean_inspection.py --download

# Offline hash and audio-contract verification.
uv run python scripts/prepare_licence_clean_inspection.py
```

All output audio is mono 16 kHz PCM16 under ignored
`data/raw/licence-clean-inspection/`. The FSD50K transport is a pinned
per-file mirror because original Freesound downloads require OAuth; official
FSD50K metadata remains the licence and attribution authority.
`licence-clean-inspection.provenance.json` records the manifest hash, class and
licence counts, revisions, and selection rules.
