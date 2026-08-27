# Gold review protocol for the 310-clip inspection

The committed manifests contain weak source or acted labels. No real inspection
row is gold. `attune.data.gold_review.GoldReviewRecord` is the strict,
append-only JSONL contract for a future human pass.

## Review procedure

Review the hash-verified local audio while showing the clip ID, waveform,
duration, source attribution, weak labels, and current Attune output. Model
predictions must not be preselected as the answer. Replay and zoom around
candidate boundaries.

Make separate decisions:

1. transcript: `accept`, `reject` with corrected text, or `not_reviewable`;
2. affect: `accept`, `reject` with one replacement category, `ambiguous`, or
   `not_reviewable`;
3. each proposed event/style: `accept`, `reject`, or `retime`; and
4. audible missing events/styles: `add`.

`retime` and `add` require millisecond onset/offset inside the clip. Follow the
first and last audible evidence. Never retain `0..duration` merely because a
source supplied only a clip tag. Use `crying_speech`/`laughing_speech` when the
behaviour modifies speech and `sob`/`laugh` for discrete events. Affect describes
perceived vocal expression, never a verified internal state.

## Ledger example

```json
{
  "schema_version": "1.0",
  "clip_id": "source-stable-id",
  "audio_sha256": "64-lowercase-hex-characters",
  "reviewer": "reviewer-id",
  "reviewed_at_utc": "2026-08-27T12:00:00Z",
  "source_label_status": "weak_source_or_acted",
  "transcript": {"decision": "accept", "corrected_text": null},
  "affect": {"decision": "ambiguous", "reviewed_label": "ambiguous"},
  "spans": [
    {
      "channel": "event",
      "source_label": "laugh",
      "decision": "retime",
      "reviewed_label": "laugh",
      "start_ms": 420,
      "end_ms": 880
    }
  ],
  "notes": "",
  "dry_run_fixture": false
}
```

The review tool copies the manifest audio hash and must fail if local audio no
longer matches. Re-review appends a later row rather than replacing history.

## Promotion rule

A single pass is reviewer-labelled, not consensus gold. Obtain an independent
second pass for every rejected, retimed, added, ambiguous, or not-reviewable
decision and for a deterministic sample of accepts. Preserve both decisions,
adjudicate disagreements separately, and report label agreement plus onset and
offset differences.

Only hash-matched rows with complete decisions and resolved adjudication may
enter a versioned gold manifest. Test identities must never be reused to fit
calibration, abstention, or model parameters.

The committed `research/error-analysis/gold-review-fixture-dry-run.jsonl`
exercises serialization and validation on generated tones/noise only. Its rows
set `dry_run_fixture=true` and are not scientific gold.
