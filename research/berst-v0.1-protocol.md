# BERSt style and affect protocol v0.1

## Purpose

Cadence still lacks release-quality shouting and external affect performance.
Existing style data are small or single-speaker, and existing affect data are
mostly clean acted recordings. BERSt adds 4,523 English clips from 98 actors in
home environments, recorded on smartphones at varied distances and
obstructions. Each raw recording groups the same actor, phrase, affect prompt,
and phone placement across ordinary, raised, and shouted delivery.

The official CC BY 4.0 source revision is pinned at
`fed35c477427cff44206464b7f85e68d1fc99062`. Its four processed Parquet shards
contain 3,503 train, 488 validation, and 532 test clips. The official partitions
contain 78, 10, and 10 speakers respectively, with no speaker overlap. Attune
maps them to `train`, `development`, and `sealed_test` without repartitioning.

An embedded-audio audit found only 4,472 distinct waveforms in the 4,523 source
rows. Thirty-one repeated-waveform groups cover 82 rows; five groups conflict
on affect, two on intensity, three on transcript, five on speaker, and one
crosses development and sealed test. Cadence excludes every row belonging to a
repeated-waveform group rather than choosing among conflicting annotations.
The prepared corpus therefore contains 4,441 unique clips: 3,429 train, 487
development, and 525 sealed test. All 98 source speakers remain represented and
the prepared partitions remain speaker-disjoint.

The source affect fields are actor prompts, not listener judgements. They are
weak intended-delivery targets and must not be described as perceived-affect
ground truth. `shout` maps only to the `shouting` style, `no-shout` is an
explicit style negative, and the three `n/a` intensity rows remain
style-unsupervised. BERSt does not supervise vocal events or whispering.

## Preparation

`scripts/prepare_berst.py` verifies the pinned README and Parquet hashes,
validates the speaker-disjoint partition, hashes every embedded waveform,
excludes every repeated-waveform group, and normalizes the retained audio to
mono PCM16 at 16 kHz. Each source `audio_id` becomes an intensity-pair group.
The verified nonsense transcript is retained for ASR replay and given a
neutral lexical-affect control label.

The BERSt test split remains unopened for model inference until a candidate is
selected using development data and the already-opened external controls. Test
label counts and source integrity may be audited, but no test prediction may
influence configuration, thresholds, checkpoints, or acceptance gates.

## Style experiment

The style run starts from the retained affect checkpoint plus whichever event
head passes the event-mixture v1.1 gates. Only `style_projection` and
`style_head` may change. Training combines BERSt train rows with the existing
style-supervised train rows using corpus-balanced sampling. Validation combines
the corresponding development rows. Whispering remains disabled regardless of
this run because BERSt contains no whisper positives.

The checkpoint is selected by development loss. Calibration is fit on
development only. It may proceed to the BERSt test split only if all of the
following pass:

- BERSt development shouting F1 is at least 0.85.
- BERSt development shouting precision and recall are each at least 0.80.
- BERSt development no-shout false-positive rate is no higher than 0.05.
- WESR shouting F1 is at least 0.55, precision at least 0.50, and recall at
  least 0.60.
- No shouting or whispering output appears on the opened 100-speaker British
  negative-control set.
- Every tensor outside the style projection/head is byte-identical to the
  initial checkpoint.

After those gates pass, the single test opening must reach shouting F1 of at
least 0.80 and no-shout false-positive rate no higher than 0.08. Gain
perturbations from -12 dB to +12 dB must not lower F1 by more than 0.05 or raise
false-positive rate above 0.10. Only `shouting` may then enter the runtime style
allowlist; `whispering` stays disabled.

## Affect experiment

The affect run is separate so its effect can be attributed. It starts from the
same accepted event/style base and updates only `affect_projection` and
`affect_head`. BERSt train/development rows are merged with the current
multi-corpus affect manifest. Sampling is balanced by corpus and within-corpus
argmax class. BERSt's intended labels are reported separately from perceptual
datasets rather than pooled into one headline score.

The affect candidate may proceed beyond development only if:

- Existing core development macro-F1 is at least 0.64.
- Acoustic Preference Score remains positive.
- BERSt development macro-F1 is at least 0.30, reflecting the difficulty of
  the source's own published baselines rather than assuming studio-corpus
  performance.
- Selective risk improves as coverage decreases on both development domains.
- No event or style tensor changes.

The already-opened RAVDESS evaluation must then reach macro-F1 of at least 0.40
without a class receiving more than 45% of predictions. If it passes, the
single BERSt test opening must reach macro-F1 of at least 0.30, show improving
selective risk, and keep the largest prediction share at or below 45%.

## Release decision

Passing BERSt alone cannot release the model. The accepted heads must be
composed, exported, recalibrated, quantized to INT8, and rerun through ASR,
event, style, affect, calibration, gain, accent, and hostile-speech controls.
The original human hostile-delivery case still requires a fresh recording.
Synthetic neutral readings remain lexical controls only.
