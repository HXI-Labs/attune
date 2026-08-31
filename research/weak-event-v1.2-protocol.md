# Weak event supervision protocol v1.2

## Purpose

The retained v0.8 event head localizes laugh, cough, and throat-clear events,
but its WESR English temporal-presence macro-F1 is 0.2882 with 0.2288 recall.
The rejected v1.1 mixture continuation reduced those values to 0.1347 and
0.1078. Version 1.2 tests whether human-recorded, weak utterance-presence
labels can improve recall without inventing timestamp targets or restoring
event hallucinations on ordinary speech.

DisfluencySpeech contributes 5,000 human-recorded English utterances from one
speaker. Its official Apache-2.0 revision is pinned at
`b7da294fe3a70dd96df6640893f2a5dfc2c87638`. The source partitions contain
4,500 training, 250 development, and 250 test clips. Attune retains those
partitions and treats them as sentence-disjoint, not speaker-disjoint. The
corpus is training and development evidence only; it cannot establish
cross-speaker or external generalization.

The source contains 665 utterances tagged with laughter, 53 with throat
clearing, 18 with sighing, and four with coughing. Tags are converted to weak
utterance-presence labels. They are not converted to frame or span labels.
The prepared manifest contains one hashed feature tensor for each of the 5,000
unique source waveforms.

## Training

The candidate starts from the retained v0.9 checkpoint and may update only
`event_head`. This is 857,877 parameters. The shared encoder, ASR path, global
event-presence head, style branch, affect branch, VAD branch, and OOD branch
must remain byte-identical.

Strong event examples retain their frame, boundary, and presence targets. Weak
examples apply a masked log-mean-exp multiple-instance loss to temporal event
logits. Positive labels encourage at least one plausible event interval;
negative labels suppress peaks across the utterance. Padding is excluded. The
weak loss does not supervise a start or end boundary.

The predeclared exploratory configuration is
`configs/training/local-mps-weak-events-v1.2.json`: at most eight epochs,
8,192 corpus-balanced samples per epoch, effective batch size 24, event-head
learning rate 3e-5, and early stopping after three non-improving development
epochs. The development loss selects the checkpoint.

## Acceptance gates

Calibration is fit on the combined development split. The candidate is
accepted only if every gate passes:

- WESR English temporal-presence macro-F1 is at least 0.34.
- WESR English temporal-presence recall is at least 0.30.
- WESR English temporal-presence false-positive rate is no higher than 0.06.
- Opened strong-label regression segment macro-F1 is at least 0.60.
- At most two of the 549 opened speech controls contain a localized event.
- DisfluencySpeech development laughter temporal-presence F1 is at least 0.65.
- DisfluencySpeech development any-event recall is at least 0.65.
- DisfluencySpeech development any-event false-positive rate is no higher than
  0.10.
- Checkpoint scope confirms that only `event_head` changed.

The DisfluencySpeech test split remains unopened because the single shared
speaker prevents it from serving as final external evidence. WESR-Bench
remains the independent external event gate and is never used for training or
threshold selection.

If any gate fails, the v1.2 checkpoint is rejected and v0.8 remains the event
head. Runtime support remains restricted to labels that pass localization and
speech-control gates; unsupported event labels stay disabled.
