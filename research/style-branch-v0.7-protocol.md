# Style branch v0.7 protocol

The v0.6 candidate uses the affect embedding for both perceived affect and
vocal style. It recognizes held-out Thorsten whisper clips, but its whisper
score is not specific enough on ordinary speech to enable in deployment. It
also has no validated shouting-speech corpus; FSD50K shouting examples are
useful acoustic positives but do not establish word-aligned shouting speech.

This experiment separates style from affect after the shared attentive pool.
The new style projection starts from the v0.6 affect projection, preserving the
candidate's initial predictions, and then adapts only through the style loss.
The SenseVoice encoder learning rate is zero, and the trainer excludes the
encoder and non-style heads from autograd. The checkpoint still serializes the
unchanged adapted encoder delta so the resulting candidate is self-contained.
ASR remains routed through the unchanged frozen base tail.

Training uses only rows with explicit style targets from the v0.7 joint
manifest. Corpus-balanced sampling gives Common Voice ordinary speech,
FSD50K style positives, and Thorsten matched ordinary/whisper speech equal
corpus weight. Eight epochs of 2,048 sampled rows are allowed, with early
stopping after three non-improving validation losses. The physical batch is 6
with four-step accumulation for an effective batch of 24. A first batch-24
attempt reached host memory pressure before its first logging interval and
wrote no checkpoint; it was stopped and excluded.

The branch is useful only if task-specific evaluation shows that a deployment
threshold can retain meaningful whisper or shouting recall without emitting
styles on ordinary speech. Affect, event, and exact-ASR gates must remain at
least as strong as the calibrated v0.6 candidate because their parameters are
not intended to change in this run. Shouting remains disabled if the available
speech supervision cannot support it.
