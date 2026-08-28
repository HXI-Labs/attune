# STARSS23 frozen MLP listen test

Research listen pack only. **STARSS23 is unwired in product infer.** Open
`index.html` from this directory in a local browser (file://). Orange bars are
the owner-authorized 100 ms STARSS23 overlays, not second-listener gold. Cyan
bars are frozen-MLP predicted spans.

Companion WAVs live in `audio/` and are gitignored. After a local train they
are copied beside the HTML so the players work. Do not public-tunnel. Do not
upload participant audio.

Regenerate after training:

```bash
export PYTHONPATH=src ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1
python scripts/train_starss23_frozen_mlp_listenpack.py \
  --sensevoice-path data/raw/model-cache/sensevoice-small
```

Metrics: `research/starss23-frozen-mlp-listenpack-results.json`. Protocol:
`research/starss23-frozen-mlp-listenpack.md`.
