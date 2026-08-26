# Phase 1 inspection set

The planned inspection set is **100–200 speaker-aware clips** for qualitative
inspection and slice checks. It is not the sealed gold baseline. No candidate
corpus listed here has been downloaded or approved for use.

## Coverage still required

The selected clips must include:

- neutral speech;
- shouting and whispering;
- crying speech, sobs, laughter, and coughs;
- Ghanaian English and British English;
- loud neutral speech and quiet high-effort speech; and
- semantic-delivery conflicts: positive text with negative delivery and
  negative text with positive delivery.

Selection must preserve source, speaker, session, licence, consent, and
annotation provenance. Speakers must not cross evaluation partitions. Human
review is still needed to confirm delivery labels, transcript polarity,
recording level, clipping, and whether acted material is suitable for each
inspection purpose.

## Candidate public sources

Every entry below has `licence_review_status: pending`. A public landing page or
stated licence is not approval; terms, subject consent, redistribution,
commercial/internal-research compatibility, attribution, and speaker metadata
must be reviewed before any download.

| Candidate | Potential coverage | Official/public source | licence_review_status |
|---|---|---|---|
| Ghana English ASR Dataset | Ghanaian English; neutral broadcast speech; possible level slices | [Ghana NLP Community on Hugging Face](https://huggingface.co/datasets/ghananlpcommunity/ghana-english-asr-2700hrs) | `pending` |
| Surrey Audio-Visual Expressed Emotion Database (SAVEE) | British English; neutral and acted emotional delivery | [University of Surrey](http://personal.ee.surrey.ac.uk/Personal/P.Jackson/SAVEE/Download.html) | `pending` |
| Dysarthric Expressed Emotional Database (DEED) | British English; neutral and elicited emotional speech; effort variation | [University of Sheffield](https://sites.google.com/sheffield.ac.uk/deed) | `pending` |
| RAVDESS | Lexically matched neutral/strong emotional delivery; candidate polarity-delivery conflicts | [Zenodo record 1188976](https://zenodo.org/records/1188976) | `pending` |
| CREMA-D | Shared sentences across emotion and intensity; candidate polarity-delivery conflicts | [Official GitHub repository](https://github.com/CheyneyComputerScience/CREMA-D) | `pending` |
| EmoV-DB | Neutral, angry, amused, and sleepy acted speech; possible intensity/effort contrasts | [Official GitHub repository](https://github.com/numediart/EmoV-DB) | `pending` |
| VocalSound | Laughter and cough events with speaker metadata | [MIT Spoken Language Systems](https://sls.csail.mit.edu/downloads/vocalsound/) | `pending` |

No single candidate is assumed to cover all slices. Loud-neutral and quiet
high-effort examples in particular require waveform-level verification and may
need a separately consented, licence-reviewed collection if public candidates
do not provide defensible examples.
