# STARSS23 first-60s gold-review protocol

This is a review protocol, not a gold claim. STARSS23 (MIT, Zenodo
[7880637](https://zenodo.org/records/7880637)) supplies human 100 ms activity
labels for class 4 laughter. That is **not** Attune gold:

- 100 ms is too coarse for a 200 ms collar;
- language is unverified (Sony Tokyo / TAU Tampere scenes);
- `laugh` versus `laughing_speech` is unsplit;
- no independent Attune review exists.

The frozen SenseVoice-frame MLP ceiling on this set remains collar F1 0.1395
versus gate 0.25. That number is a failed model result, not a review target:
do not compare gold collar to 0.1395. This pack does not unfreeze the encoder,
train a head, tile scenes, or wire STARSS23 into the decoder.

## Pack

`data/manifests/starss23-gold-review-pack.jsonl` is the 49 first-60s inspection
rows (`source_window_start_ms == 0`) copied from the unmerged PR 22 manifest
`data/manifests/starss23-scene-raster-inspection.jsonl` on
`cursor/starss23-scene-timing`. It contains 48 source laugh events and keeps
the 29 zero-laugh clips as true negatives. Later 60 s tiles are excluded:
tiling was a failed train protocol, not a review unit.

Three source laughs end at the 60 s window cut (even though PR 22 stored
`clipped_spanning_event_count = 0` on first tiles). They are flagged
incomplete and collar-ineligible. Do not gold clip-end as offset:

- `fold4_room24_mix006-w000000` 59400–60000
- `fold4_room16_mix010-w000000` 51900–60000
- `fold4_room8_mix002-w000000` 59500–60000

Pack JSONL source `start_ms` / `end_ms` stay the STARSS23 100 ms overlays.
The pack builder never overwrites them.

Each row keeps its committed SHA-256, 60 s duration, `mean_of_4_tetrahedral_mic`
downmix, 16 kHz mono contract, and
`label_status: STARSS23 human activity label; not reviewed Attune gold`.
Ledger rows for this pack must set

`source_label_status = human_100ms_activity_not_attune_gold`

They must not reuse `weak_source_or_acted` (that value remains for the 310-clip
acted/weak inspection dry-run).

## Review procedure

1. Hash-verify local wavs with `scripts/review_starss23_gold.py`. A SHA-256
   mismatch fails closed. Do not invent, resample, or substitute audio.
2. Open the gitignored HTML pack under
   `artifacts/gold-review/starss23-first60s/`. Hear the full 60 s clip. Zoom
   around each source span. Source 100 ms (merged) activity is shown as an
   overlay only.
3. Do **not** preselect Attune model predictions as the answer. This pack does
   not display model scores.
4. Source 100 ms spans are overlays, not onset gold. The UI does not offer
   `accept`. Decide `reject`, `retime` (default), or `add` at free millisecond
   resolution. Never snap to 100 ms. Onset is the first voiced burst; offset is
   the last voiced frame. Split if a silent/unvoiced gap is >= 300 ms. Label
   only what is downmix-audible.
5. Split `laugh` (discrete non-speech event) versus `laughing_speech` (laughter
   modifying speech). Source class 4 is unsplit; the reviewer must choose.
6. Do not review transcript or affect. STARSS23 consent is not independently
   verified; do not transcribe overlapping speech. The ledger auto-fills those
   fields as `not_reviewable`.
7. `add` an event or style when audible laughter is missing from the source
   spans. Multiple source laughs on one clip are identified as
   `laugh@{start_ms}-{end_ms}` so the append-only ledger stays unique.
8. Privacy: STARSS23 is public MIT natural scenes with identifiable speech.
   Consent is not independently verified. Review stays on a local machine.
   Never commit wavs. Never upload participant audio.

The playable tool writes `GoldReviewRecord` rows (schema_version 1.0) to the
gitignored append-only ledger
`artifacts/gold-review/starss23-first60s/ledger.jsonl`. Re-review appends; it
does not rewrite history. The review tool copies the pack hash into the row and
must refuse a mismatched hash.

## Promotion rule

Unchanged from `docs/gold-review-protocol.md`:

- one pass is reviewer-labelled, not gold;
- an independent second pass is required;
- preserve both decisions, adjudicate disagreements separately;
- only hash-matched rows with complete decisions and resolved adjudication may
  enter a versioned gold manifest.

`attune.data.gold_review.evaluate_gold_promotion` never marks gold from a
single pass, from dry-run fixtures, or without explicit adjudication. This PR
does not produce gold. The scientific gate stays closed until those passes
exist.

## Local commands

```bash
# Rebuild the 49-row pack from a checkout of the unmerged PR 22 inspection JSONL.
uv run python scripts/build_starss23_gold_review_pack.py \
  --inspection-manifest /path/to/starss23-scene-raster-inspection.jsonl

# Hash-verify local first-60s wavs and write the gitignored HTML pack.
uv run python scripts/review_starss23_gold.py \
  --cache-dir data/raw/starss23-scene-raster/inspection_test

# Optional: serve locally so Save appends ledger.jsonl. Do not expose.
uv run python scripts/review_starss23_gold.py --serve
```

Prefer the first-60s non-tiled cache (hash-matched numbered wavs). The tiled
cache is a different protocol; it may be used only when the SHA-256 of a
first-60s row still matches. A failed hash stops the run.
