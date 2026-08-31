# Event calibration v0.9 protocol

The weak-event v1.2 checkpoint was rejected. At its fixed runtime confidence
floor it reached WESR localized-presence F1 of 0.0235 and opened-regression
segment F1 of 0.0345. A threshold sweep could not recover the predeclared
quality requirements, so the checkpoint is not a release candidate.

The retained event-hardening v0.8 head remains the stronger model. Its original
global 0.98 confidence floor produced WESR F1 of 0.2882, recall of 0.2288, and
false-positive rate of 0.0491, while retaining opened-regression segment F1 of
0.6204 with no false-positive clips across 549 speech controls.

## Selected calibration

WESR has already been opened and is treated as a selection set, not as an
untouched final test. A label-specific threshold sweep selected:

- `laugh`: 0.83;
- `cough`: 0.975;
- `throat_clear`: 0.98.

All other localized labels remain disabled. Event-presence outputs remain
disabled. The selected calibration reaches WESR localized-presence F1 0.3631,
recall 0.3007, and false-positive rate 0.0596. Opened-regression segment F1 is
0.6622 and the 549 speech controls still contain zero localized false-positive
clips.

These are post-selection measurements and do not establish release quality.
The calibration may advance only if it passes one untouched evaluation on the
sealed Disfluency Speech split with laugh F1 and recall at least 0.65,
false-positive rate at most 0.10, and no regression-control degradation. The
fresh consented human hostile-speech recording remains a separate mandatory
release gate. If either confirmation fails, the calibration is rejected.
