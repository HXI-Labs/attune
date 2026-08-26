# Synthetic semantic-conflict fixtures

These tiny mono PCM16 WAV files are generated tones/noise, not recordings of
people and not a scientific gold set. They exist only to test evaluation
wiring, schema validation, baseline execution, and Acoustic Preference Score
(APS). Do not report these numbers as evidence of real-world affect accuracy.

`manifest.json` records each transcript, acoustic (delivery) target, lexical
target, and fixture group. `gold.json` contains a complete schema-valid
`AttuneOutput` for every item. The synthetic acoustic target is encoded only by
the fixture metadata; the simple tones are not claimed to naturally convey the
named affect.

The cases cover:

- neutral text with varying delivery labels;
- explicit text whose lexical and delivery labels match;
- explicit text conflicting with its delivery label; and
- same-text/different-delivery pairs.

Regenerate the WAVs deterministically with:

```bash
python data/fixtures/semantic_conflict/generate_audio.py
```
