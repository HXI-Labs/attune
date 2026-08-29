# Human-interaction roadmap after v0.1

v0.1 does not train on newly collected human interaction preferences and does
not claim downstream conversational benefit. The next evidence stage should
use public or consented audio scenarios while collecting response judgments,
not new covert voice recordings.

## Proposed study

- Freeze the Attune model and downstream language model, prompt, decoding
  settings, and conversation history.
- Compare transcript-only, hard-label, calibrated structured context, and
  original-audio frontier-reference conditions.
- Randomize and blind 60–100 scenarios. Include neutral controls, abstentions,
  semantic/acoustic conflict, false-positive tags, and observable distress.
- Collect pairwise preference plus appropriateness, understanding, restraint,
  patronising, overreaction, underreaction, and safety ratings.
- Pre-register the main endpoint: structured calibrated context versus
  transcript only. Report confidence intervals and neutral-speech non-inferiority.

## Optional preference optimization

Only after the blinded study shows a stable preference signal, convert
consented comparisons into a separate downstream-policy dataset. Train a small
prompt/policy adapter with preference optimization while keeping Attune frozen.
The speech model must not learn conversational rewards that encourage stronger
emotion assertions. Maintain held-out neutral and incorrect-metadata scenarios,
and reject an adapter that increases patronising or overreactive responses.

This stage trains response policy, not a claim to recover a speaker's internal
state. It remains outside the v0.1 definition of done.
