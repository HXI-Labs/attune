# Phase 1 baseline runners

All runners implement `BaselineAdapter` and return a schema-valid
`AttuneOutput`. Unsupported channels are not inferred:

- Whisper and transcript-only runners emit empty `styles` and `events`.
- SenseVoice maps only recognized rich-transcription/AED tags onto the Attune
  ontology. Unmapped tags such as noise, music, or applause are dropped.
- Placeholder word timings divide the clip uniformly and use confidence `0.5`;
  they are not alignment results.
- Unsupported affect dimensions use value `0.0` and confidence `0.0`.
- Unknown quality probabilities are Phase 0 placeholders.
- An explicitly configured `StubEventHead` emits no events. Its zero event
  scores are expected, not a claim that fixtures contain no events.

The transcript-only lexicon runner is deterministic, CPU-only, and always
available. It receives a supplied transcript and is intentionally sensitive to
lexical content; the semantic-conflict fixtures expose that limitation through
APS.

Whisper-Small, SenseVoice-Small, and emotion2vec+ are lazy optional adapters.
They never initiate downloads. Each requires a local path environment variable
or an existing recognized Hugging Face cache snapshot plus its Python runtime:

```bash
uv sync --extra model-runners
```

| Runner | Local path variable | Runtime |
|---|---|---|
| Whisper-Small | `ATTUNE_WHISPER_SMALL_PATH` | `torch`, `transformers` |
| SenseVoice-Small | `ATTUNE_SENSEVOICE_SMALL_PATH` | `funasr` |
| emotion2vec+ | `ATTUNE_EMOTION2VEC_PLUS_PATH` | `funasr` |

SenseVoice additionally requires `ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1`. Set it
only after the separate weight licence review is recorded in
`data/provenance/sensevoice_small_weights.yaml`. CI must not download weights.
The 2026-08-26 records permit official SenseVoice-Small, Whisper-Small, and
emotion2vec+ weight downloads for internal baseline runs only. They do not
authorise fine-tuning, public weight redistribution, third-party conversions,
or MSP-Podcast use. See `data/provenance/` for exact terms and attribution.

SenseVoice's off-the-shelf tags map as follows: laughter to `laugh`, cry/crying
to `sob`, sigh to `sigh`, cough to `cough`, throat clearing to `throat_clear`,
sneeze to `sneeze`, and breath/breathing to `breath`. Singing, whispering, and
shouting tags map directly to styles. Laughter or crying tags accompanied by a
non-empty transcript also produce `laughing_speech` or `crying_speech`; this is
a low-confidence utterance-level heuristic, not proof that every spoken word
has that style.

Current SenseVoice AED output has no event score or frame boundaries. Such
annotations therefore use confidence `0.0`, status `provisional`, and span the
whole clip. A structured runtime result with an explicit score preserves that
score. These spans are **not frame-level localization**. Duplicate tags collapse
to one ontology label per clip.

`ModularCascade` combines an ASR adapter and an affect adapter while retaining
events and styles emitted by the ASR source. Consequently the SenseVoice
cascade uses SenseVoice AED output; the Whisper cascade still emits no events.
An explicit `StubEventHead` can override ASR events when a no-op head is needed.
This replaceable composition is the Phase 1 system; it is not a unified or newly
trained model.
