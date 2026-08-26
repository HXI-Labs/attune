# Ethics and use constraints

Attune estimates how a voice is perceived and which vocal behaviours are
observable. It does not know internal emotion, diagnose illness, detect
deception, or justify consequential decisions.

## Prohibited uses

- covert monitoring, surveillance, or analysis without meaningful consent
- automated or materially influential decisions in hiring, employment,
  lending, insurance, policing, criminal justice, education, or housing
- medical or mental-health diagnosis, triage, risk scoring, or treatment
  decisions
- deception, truthfulness, intent, guilt, or credibility detection
- inference of protected traits, identity, disability, health status, or other
  sensitive attributes
- punitive profiling, manipulation, or targeting of vulnerable people

Attune outputs must not be used as evidence that a person feels a particular
emotion or has a condition.

## Required output language

Describe perception and uncertainty: **“your voice sounds strained”** or “the
recording has a 0.61 probability of strained speech.” Never state **“you are
depressed,”** “you are angry,” “you are lying,” or equivalent claims about an
internal state, diagnosis, or intent.

Interfaces must expose confidence, abstention, scope, quality limitations, and
the warning: “Vocal affect is a probabilistic perception, not a verified
internal state.” Spoken transcript and metadata remain separate trusted
channels so labels cannot be mistaken for user-authored text.

## Operational requirements

Obtain consent, minimize retention, protect raw audio, document data licences,
test speaker-disjoint performance and calibration, examine subgroup and
recording-condition failures, and provide a way to disable or delete analysis.
Human review does not make a prohibited high-stakes use acceptable.
