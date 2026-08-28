# Partial cascade inventory (2026-08-28)

Search for a live DCASE-stamped wav demo. Gold gate remains closed. STARSS23
natural gold is paused. No Cloud Agents. No merge.

## DCASE gated frame-head

Not found. Expected path `artifacts/dcase-frame-localization/frame-head.pt` is
absent, as are `artifacts/dcase-localization/frame-head.pt` and
`temporal-head.pt`.

The head was trained in Cloud Agent `bc-05c2a10e-5c83-4ffa-befa-e8290ebbc8ec`
(`scripts/train_temporal_localization.py`, seed 0, 30 epochs, 66,051-param
`512→128→3` MLP on frozen `sensevoice-small-encoder-frames-v1`). The recorded
held-out result is in `research/dcase-frame-localization-results.json`
(hysteresis segment/collar F1 0.7059/0.5279 vs whole-clip 0.3183/0). The
checkpoint was gitignored and did not survive onto this box.

Local reproduction needs existing DCASE caches. They are not here:

- `data/raw/dcase2016-task2/` missing (archives 125,450,464 and 328,618,451 bytes)
- `data/raw/dcase2016-localization/` missing (216 development + 100 inspection wavs)
- `artifacts/dcase-frame-localization/embeddings/` missing

This work does not download those corpora and does not fake timestamps.

## STARSS23 heads (refused)

Present under `artifacts/starss23-scene-raster/` and unwired. SHA-256:

| path | sha256 |
|---|---|
| `frame-head.pt` | `e1dc3796a8084ca7b7231db0ca7ea61f30aed6779e8fa7af6c25b265bdc58776` |
| `frame-head-40epoch.pt` | `5374e598f5d1007b84c0805ab1b3ec51f8b2f421587743cf7429bc04ab6e002a` |
| `frame-head-mlp-v2.pt` | `1181efea244a4e1b13991c3c0a956d05649210fbed46707a6c4e30f17ccd3c38` |
| `frame-head-tiled.pt` | `f1145e5149982fc2aac63e86cb6075a9031baeec7c5def002b68ef2ef99531c6` |
| `frame-head-tiled-valfirst60s.pt` | `33f2024cf0a6962ebb177998d654cda088c26b1eb593f0dc3ac1806ef9ba493b` |

`scripts/infer.py` refuses these as `--temporal-head` and never auto-selects them.

## What the wav CLI can actually run on this box

| Artifact | On disk |
|---|---|
| SenseVoice-Small | yes — `data/raw/model-cache/sensevoice-small/model.pt` sha256 `833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea` |
| emotion2vec+ | no |
| VocalSound `artifacts/event-probe/head.pt` | no |
| FSD50K `artifacts/fsd50k-event-probe/head.pt` | no |
| DCASE frame-head | no |

`scripts/infer.py` therefore runs SenseVoice AED tags only, abstains affect, and
labels the HTML as utterance-scope vs missing DCASE timestamps. That is not a
timing demo and not a full Phase 2 cascade. Inspection evaluation still needs
the complete probe package.


## Live HTML that exists

`research/demo/sensevoice-only.attune.html` is a playable SenseVoice-only
timeline (words from the official English example). It labels missing DCASE
timestamps and abstained affect. It does **not** stamp laugh/cough/throat_clear.
The companion WAV is gitignored.
