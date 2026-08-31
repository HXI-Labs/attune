# Event hardening v0.8 protocol

The v0.6 temporal head learned localized positive spans from DCASE, but ordinary
speech controls supplied only utterance-level event-presence negatives during
training. The frame head therefore never received an explicit whole-clip
negative loss on speech. It later emitted one false laugh and three false
throat-clear spans across 1,029 opened speech-only controls.

Version 0.8 marks validated Common Voice, CREMA-D, and Thorsten speech as empty
localized-event clips. Existing features and partitions are reused. DCASE
retains its positive spans, and all other supervision is unchanged. This is
weak negative annotation: a rare incidental event may be missed, so the
protocol does not extend the policy to uncontrolled corpora by default.

The generated manifest has SHA-256
`63c1d59e63028918ffc4868b33e4f4de695ed80800ab0ebbf87000fe1750b42d`.
It contains 11,784 unique feature files and 3,453 localized-event speech
controls. Its integrity audit reports no overlap among speaker-disjoint
corpora. Thorsten is explicitly sentence-disjoint, with all 300 sentence-pair
groups confined to one partition; this policy had been present in the source
schema but was restored to the prepared-feature schema before this run.

Starting from the completed style-branch candidate, training updates only the
temporal event head. The shared encoder, ASR route, affect branch, style branch,
and all other heads remain unchanged and are serialized into the resulting
self-contained delta. Corpus-balanced sampling prevents the substantially
larger ordinary-speech set from removing event recall.

Acceptance requires zero localized-event clips on development speech controls,
no opened-regression speech-control false positives, and no more than two
absolute points of segment macro-F1 loss from the v0.6 opened-regression result
of 0.6160. The fresh British confirmation set remains sealed until affect,
style, and event gates pass together.

## Result

Early stopping selected epoch 3 at validation loss 0.000705. The run trained
857,877 temporal-head parameters. Its checkpoint has SHA-256
`ec87b3c204f5d8c791880500e63e99fc559fd3df155e5751d2f49f9aa9327a00`;
tensor comparison found ten changed tensors, all under `event_head`. The ONNX
export has SHA-256 `a2281dac4643cb90f4fa5959d84d8b89adf1c081fbd741ea78ed3f9772326dd4`
and maximum ten-output parity error `6.64e-05`.

Development localized segment macro-F1 is 0.9188 with zero false-positive
clips across 542 localized-event speech controls. The frozen-calibration
opened regression result is 0.6204, slightly above the 0.6160 reference, with
zero false-positive clips across 549 controls. All 480 opened RAVDESS clips
also remain free of localized false positives. Affect and APS are unchanged.

The experiment therefore passes its predeclared hardening acceptance. It does
not establish broad event generalization: on the external WESR English set,
whole-clip presence derived from the temporal decoder has macro-F1 0.2882,
22.9% recall, and a 4.9% false-positive rate on clips without the three enabled
localized classes. The global event-presence head remains disabled, and the
model remains unreleasable pending broader positive training data.
