# STARSS23 Jerry first-pass listen (not gold)

One-pass reviewer labels on 6 first-60s STARSS23 inspection clips. This is
**reviewer_labelled**, not Attune gold. The encoder stays frozen. STARSS23
timestamps stay unwired. The 0.25 collar gate is unchanged. No number here
replaces reported best frozen-MLP collar F1 **0.1395** (48-event first-60s
control).

## N

- 6 clips reviewed (4 laugh clips + 2 true negatives)
- 9 retimes, 1 reject, 2 true-negative clips
- Ledger: `research/error-analysis/starss23-jerry-firstpass-ledger.json`
- Source 100 ms overlays remain `human_100ms_activity_not_attune_gold`

## 100 ms vs Jerry (onset-only, not a model score)

Jerry retimes are the reference; STARSS23 100 ms onsets are the prediction.
200 ms onset collar, offset ignored.

- TP 1 / FP 9 / FN 8
- precision 0.10, recall 0.1111, **F1 0.1053** (~0.11)
- 1/9 onsets within 200 ms; median |onset| 392 ms

## Frozen 40-epoch MLP vs this subset (ran)

Checkpoint `artifacts/starss23-scene-raster/frame-head-40epoch.pt`
(SHA-256 `5374e598…`, 0a27733 family). Locked decoder: high 0.95, low 0.855,
gap 0, min_active 1, median 3. Mean-of-4 first-60s embeddings reused
(6 cache hits, encoder not run, no extract, no train, no grid).

Inspection-style onset+offset collar on **only these 6 clips**:

| | gold | collar F1 | TP / FP / FN | segment F1 |
|---|---|---:|---|---:|
| **A** (control) | STARSS23 100 ms | **0.0** | 0 / 20 / 10 | 0.5161 |
| **B** | Jerry retimes (reject = no event; TNs scored) | **0.0** | 0 / 20 / 9 | 0.4762 |

A does **not** resemble the 0.1395 regime: this is a 6-clip slice, not the
48-event control. Both TNs stayed silent (max_prob 0.24 / 0.22). Model bursts
are short; several STARSS23 onsets fall inside 200 ms if offset is ignored
(onset-only A F1 0.40, not a gate number), but Jerry onsets are later
(onset-only B F1 0.0). Neither figure replaces 0.1395.

## Next lever

More onsets, or an independent second listener, then adjudication. Do not
unfreeze the encoder, grid the decoder, lower the 0.25 collar gate, or wire
STARSS23. Gold stays closed until two passes exist.
