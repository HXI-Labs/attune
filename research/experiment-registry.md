# Experiment registry

Every run receives an immutable ID and a linked configuration/commit. Do not
record results only in notebooks.

| Run ID | Date | Commit | Data + manifest hash | Model + licence | Split | Seed | Metrics | Artifacts | Decision |
|---|---|---|---|---|---|---:|---|---|---|
| phase2-vocalsound-frozen-v1 | 2026-08-26 | `73dced9c5121254252ea7f096f252f0d9eacabec` | VocalSound speaker-disjoint protocol; partition identities in report | SenseVoiceSmall `3847d57`, FunASR Model License v1.1; linear head | speaker-disjoint train/validation/80 inspection | 0 | inspection macro-F1 0.8598 | `research/sensevoice-frozen-probe-metrics.json` | Keep frozen probe; no localization claim |
| phase2-fsd50k-frozen-v1 | 2026-08-27 | `f071e6f7f94d5ba918ba4c2b70d0ecb31258351c` | `fsd50k-frozen-probe.jsonl` `731bd387…` | SenseVoiceSmall `3847d57`, FunASR Model License v1.1; linear head; source clips CC0/CC BY | clip-disjoint train/validation/100 inspection | 0 | inspection macro-F1 0.7410 | `research/fsd50k-frozen-probe-metrics.json` | Keep frozen probe; standalone labels only |
| phase1-cascade-calibrated-310 | 2026-08-27 | `fde5fc4ed86d99a1eb1c263734e80ba77e988980` | original `7c805b27…`; expansion `f55876ba…` | SenseVoiceSmall + emotion2vec+ under FunASR model agreement; gitignored probe heads | 310 untouched inspection clips | 0 | WER/F1/OOD/calibration in report | `research/attune-cascade-inspection-results.json` | Gate closed |
| phase1-dcase-temporal-negative | 2026-08-27 | `d0b7544286b187e50eecdcec0e004f4de2f97705` | hashed 168/48/100 DCASE protocol | frozen SenseVoiceSmall + 66,051-param temporal MLP | development train/validation; source-disjoint public test | 0 | segment F1 0.1943 vs 0.3183 whole-clip; collar F1 0 | `research/dcase-localization-results.json` | Reject temporal head; utterance scope remains |
| phase2-affect-abstention-v1 | 2026-08-27 | `0b0c97a72a88e04b7b3720f0a88651d106951ebd` | validation `affect-calibration.jsonl` `b6d6a495…`; original/expansion manifests above | emotion2vec+ base `b318240`, FunASR Model License v1.1; no training | 120 actor-disjoint validation; 130 untouched acted inspection | 0 | threshold 0.8337; test coverage 0.7615; retained F1 0.9819; full-population F1 0.8372 | `configs/calibration/phase2.json`; `research/phase2-probes-summary.json` | Operational selective output; gate closed |
| timing-frame-dcase-v1-blocked | 2026-08-27 | this PR | existing hashed 216-development/100-public-test DCASE protocol; local audio cache absent | frozen SenseVoiceSmall frames, FunASR Model License v1.1; 66,051-param MLP | scene/file-disjoint development train/validation; source-disjoint public test | 0 | blocked before extraction; no frame metric | `research/timing-holes-results.json` | Do not wire timestamps; gate closed |

For each run, record hardware, preprocessing, ontology/schema versions,
checkpoint provenance, hyperparameters, calibration method, abstention
threshold, runtime/memory, subgroup and acoustic slices, failures, and whether
results were reproduced. Exploratory notebook findings must be promoted into a
scripted run before they count as evidence.
