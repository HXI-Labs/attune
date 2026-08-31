# Cadence compact cascade v0.1 protocol

This protocol fixes the release-candidate composition and acceptance checks
before the integrated candidate is scored. Its purpose is to replace repeated
shared-encoder fine-tuning with independently gated acoustic components.

## Fixed composition

- Cadence affect-focus v0.9 remains the 241,904,650-parameter ONNX graph.
- The frozen base-ASR view supplies CTC transcription and a padding-safe
  5,120-value probe embedding in one encoder pass.
- A VocalSound linear head may add `laugh`, `sigh`, `cough`, `throat_clear`, or
  `sneeze` at utterance scope.
- Styles are disabled in v0.1. A BERSt binary shouting head was evaluated but
  missed the predeclared development gate (F1 0.689, recall 0.551) at the
  required ordinary-speech false-positive limits. `whispering` is also disabled
  until a speech-specific head passes an equivalent protocol.
- The event head uses a trained `none` logit and abstains unless its validation-set
  margin threshold is met. Raw SenseVoice AED and emotion tags are excluded.
- Existing joint event-presence and style allowlists remain empty. The retained
  v0.9 localized-event branch remains limited to `laugh`, `cough`, and
  `throat_clear` at confidence 0.98 or above.
- emotion2vec+ remains an evaluation and teacher baseline. It is not part of
  the compact deployment model.

The linear event head adds fewer than 31,000 parameters. The unique deployment
parameter count therefore remains below 242 million. The NumPy head artifact
do not require PyTorch at inference time.

## Opened development checks

The existing VocalSound inspection partition and BERSt development partition
are opened benchmarks,
not sealed gold data. They may determine whether a head is worth carrying into
the release candidate, but they cannot support a final generalization claim.

The VocalSound head must reach macro-F1 of at least 0.80. The BERSt head must
reach shouting F1 of at least 0.85 with precision and recall of at least 0.80,
and a false-positive rate no greater than 0.05 on BERSt ordinary speech. It
must emit no style on the external ordinary-speech controls. A head that misses
its gate is omitted rather than compensated for by lowering a threshold.

The discarded FSD50K head is not a release component. It reached 0.735 macro-F1
on its opened sound-event inspection set but falsely labelled all four ordinary
hostile-lexical speech controls as `whispering`. This is recorded as a rejected
experiment, not repaired by changing its threshold.

## Integration checks

The candidate must satisfy all of the following:

1. The exported ONNX probe embedding matches the frozen training representation
   within a maximum absolute error of 0.0005 on at least 20 clips.
2. Each enabled head has a false-positive clip rate no greater than 0.05 on
   speech controls outside its source domain.
3. The four neutral hostile-lexical controls produce an exact normalized
   transcript, no event or style output, and affect abstention.
4. JSON schema validation, deterministic XML, batch API, and pseudo-streaming
   smoke checks pass.
5. INT8 is evaluated only after the full-precision candidate passes these
   checks. Probe decisions must agree between full precision and INT8 on the
   release regression suite.

The earlier affect cross-corpus blocker and the required fresh consented human
hostile-speech recording remain independent release gates. Passing this
protocol produces a private release candidate, not permission for a public
weight release.
