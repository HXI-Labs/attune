# BERSt style v0.2 protocol

## Motivation

The frozen-encoder v0.1 candidate was rejected before the BERSt sealed split
was opened. It reached BERSt development shouting F1 of 0.8604, but its
no-shout false-positive rate was 0.2292. On external WESR, F1 was 0.4855 and
precision was 0.3652. Threshold analysis showed that calibration could not meet
the false-positive and recall requirements simultaneously.

The v0.1 run also exposed an implementation error: target-specific training
froze every shared encoder parameter after the adaptation policy had selected
the upper two blocks. The configured encoder learning rate therefore had no
effect. V0.2 corrects that behavior and tests the partial-adaptation hypothesis
that was already part of the project plan.

## Predeclared changes

V0.2 starts from the retained v0.1 style checkpoint. It may update only:

- `style_projection` and `style_head`;
- SenseVoice encoder blocks 47 and 48;
- the perception path's `after_norm` and `tp_norm` parameters.

The copied base-ASR tail remains frozen. Training uses an encoder learning rate
of 1e-6, a head learning rate of 2e-5, eight permitted epochs, effective batch
size 32, seed 43, and early stopping after three non-improving epochs. The style
focal loss uses positive alpha 0.35 instead of the sparse-event default 0.75.
This gives no-shout examples more influence without changing event training.

No BERSt sealed predictions may be produced during development. The already
opened BERSt development, WESR, and British control sets may be used for model
selection under the existing fixed requirements.

## Development gates

The candidate may open BERSt sealed data only if all of these pass:

- BERSt shouting F1 is at least 0.85.
- BERSt shouting precision and recall are each at least 0.80.
- BERSt no-shout false-positive rate is at most 0.05.
- WESR shouting F1 is at least 0.55.
- WESR shouting precision is at least 0.50.
- WESR shouting recall is at least 0.60.
- The opened British negative-control set produces no style false positives.
- The checkpoint-scope audit finds no change outside the declared style and
  perception-encoder parameters.

If the gates pass, the one-shot BERSt sealed requirements remain F1 at least
0.80 and no-shout false-positive rate at most 0.08. The existing gain-sweep
requirements also remain unchanged. V0.2 must be rejected if shared adaptation
damages the final event, affect, calibration, or ASR gates, even when its style
metrics pass.
