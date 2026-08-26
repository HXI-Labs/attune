# Experiment registry

Every run receives an immutable ID and a linked configuration/commit. Do not
record results only in notebooks.

| Run ID | Date | Commit | Data + manifest hash | Model + licence | Split | Seed | Metrics | Artifacts | Decision |
|---|---|---|---|---|---|---:|---|---|---|
| _none_ | | | | | | | | | |

For each run, record hardware, preprocessing, ontology/schema versions,
checkpoint provenance, hyperparameters, calibration method, abstention
threshold, runtime/memory, subgroup and acoustic slices, failures, and whether
results were reproduced. Exploratory notebook findings must be promoted into a
scripted run before they count as evidence.
