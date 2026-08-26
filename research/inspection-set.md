# Phase 1 inspection set

The planned inspection set is **100–200 speaker-aware clips** for qualitative
inspection and slice checks. It is not the sealed gold baseline. The licence
review recorded on 2026-08-26 is an operational research review, not legal
advice or lawyer sign-off. No corpus was downloaded as part of this review.

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

The statuses below govern the current research phase. They do not by themselves
approve redistribution, commercial use, or inclusion in a released training
set. Audio must remain off git, and CI must not download restricted or
registration-gated corpora. The per-corpus ledgers in `data/provenance/` record
the reviewed terms and remaining conditions.

| Candidate | Potential coverage | Official/public source | licence_review_status |
|---|---|---|---|
| Ghana English ASR 2700hrs | Ghanaian English; neutral speech; possible level slices | [Ghana NLP Community on Hugging Face](https://huggingface.co/datasets/ghananlpcommunity/ghana-english-asr-2700hrs) | `reviewed_nc_only` |
| Surrey Audio-Visual Expressed Emotion Database (SAVEE) | British English; neutral and acted emotional delivery | [University of Surrey registration](https://cvssp.org/savee/Register.html) | `pending_registration` |
| Dysarthric Expressed Emotional Database (DEED) | British English; neutral and elicited emotional speech; effort variation | [University of Sheffield](https://sites.google.com/sheffield.ac.uk/deed) | `pending_access_request` |
| RAVDESS | Lexically matched neutral/strong emotional delivery; candidate polarity-delivery conflicts | [Zenodo record 1188976](https://zenodo.org/records/1188976) | `reviewed_nc_only` |
| CREMA-D | Shared sentences across emotion and intensity; candidate polarity-delivery conflicts | [Official GitHub repository](https://github.com/CheyneyComputerScience/CREMA-D) | `reviewed_internal_research_ok` |
| EmoV-DB | Neutral, angry, amused, and sleepy acted speech; possible intensity/effort contrasts | [Official GitHub repository](https://github.com/numediart/EmoV-DB) | `pending` |
| VocalSound | Acted laugh, sigh, cough, throat-clear, and sneeze events with speaker metadata; not inline speech | [MIT Spoken Language Systems](https://sls.csail.mit.edu/downloads/vocalsound/) | `reviewed_internal_research_ok` |
| MSP-Podcast | Naturalistic emotional speech | [UT Dallas MSP Lab](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) | `pending` |

No single candidate is assumed to cover all slices. Loud-neutral and quiet
high-effort examples in particular require waveform-level verification and may
need a separately consented, licence-reviewed collection if public candidates
do not provide defensible examples.

## Collection constraints and next step

- `reviewed_nc_only` sources may be used for this non-commercial research phase,
  but must not enter a commercially redistributed training set without a paid
  or otherwise applicable commercial licence. Ghana English ASR must be
  replaced or re-licensed before any commercial Attune release.
- Jerry must register for SAVEE. DEED access requires an email request to
  `dysarthricdeed@gmail.com`. Neither source may be downloaded in CI or
  redistributed.
- EmoV-DB remains blocked until its exact weights/data licence file is located
  and quoted. MSP-Podcast remains pending.
- The recommended next collection, once access conditions are satisfied, is a
  small local subset of VocalSound events plus speaker-disjoint CREMA-D
  same-sentence pairs for semantic-delivery conflict inspection. Keep audio
  local and off git; only manifests and hashes may be committed later.
- The 14-day no-large-fine-tuning gate remains in force.
