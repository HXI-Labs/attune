# DCASE last-block encoder iterates (locked decoder)

Decoder locked: exact 0.95; hysteresis 0.95 / 0.855 / gap 1 / min 1. Gate write-once: exact ≥ 0.4637, hysteresis ≥ 0.5279, segment margin ≥ 0.05. Iterate 1–2 last block `encoder.tp_encoders.19`. Iterate 3 last two blocks `encoder.tp_encoders.18` and `.19`. Head init SHA `cb74b1d493511d384dea3abf37139cf754a16f540b594b7f92f82647204b01fc`. Seed 0. CPU. STARSS23 unwired.

Iterate 1 (`encoder-lr 2e-5`, `head-lr 1e-4`) is frozen evidence and was not rerun.

| variant | exact | hysteresis | margin | val peak | decision |
|---|---:|---:|---:|---:|---|
| iterate 1: enc 2e-5 / head 1e-4, 9/10 ep, pat 3 | 0.4755 | 0.5169 | +0.4059 | 0.6891 | keep wired head (hyst miss) |
| iterate 2.1: enc 5e-6 / head 3e-5, 10/10 ep, pat 3 | 0.4910 | 0.5000 | +0.4070 | 0.6154 | keep wired head (hyst miss) |
| iterate 2.2: enc 5e-6 / head 1e-4, 11/15 ep, pat 5 | 0.4911 | 0.5247 | +0.4003 | 0.6897 | keep wired head (hyst miss; closest) |
| iterate 2.3: enc 5e-6 / head 3e-5 / enc-wd 0.05, 12/15 ep, pat 5 | 0.4910 | 0.5000 | +0.4070 | 0.6154 | keep wired head (hyst miss) |
| iterate 3.1: last two blocks enc 5e-6 / head 1e-4, 11/15 ep, pat 5 | 0.4929 | 0.5247 | +0.4007 | 0.7059 | keep wired head (hyst miss) |
| iterate 3.2: last two blocks enc 5e-6 / head 3e-5 / enc-wd 0.05, 15/15 ep, pat 5 | 0.4610 | 0.5019 | +0.4152 | 0.6271 | keep wired head (exact+hyst miss) |

JSON: `research/dcase-encoder-lastlayer-results.json`, `research/dcase-encoder-lastlayer-lr5e6-results.json`, `research/dcase-encoder-lastlayer-lr5e6-head1e4-results.json`, `research/dcase-encoder-lastlayer-wd-results.json`, `research/dcase-encoder-last-two-block-results.json`, `research/dcase-encoder-last-two-block-wd-results.json`. Weights stay gitignored.

Variant 2.3 selected the same val-best epoch as 2.1 (epoch 7, val hyst 0.6154); inspection collar numbers matched 2.1. Last-block weight decay 0.05 at this LR did not move discrete event F1.

Iterate 3 unfroze `encoder.tp_encoders.18` and `.19` (6,316,032 encoder params). Variant 3.1 matched 1-block 2.2 hysteresis at 0.5247 and still missed 0.5279. Variant 3.2 also missed exact (0.4610 < 0.4637). Two-block search stops here.

**Conclusion:** last-block LR/WD and last-two-block (2.2 hyperparams + one WD follow-up) are exhausted on this locked decoder. Keep the PR 26 frozen head wired. Do not merge. Do not touch STARSS23. Do not lower the gate. Do not grid-search the decoder.
