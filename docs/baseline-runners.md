# Phase 1 baseline runners

All runners implement `BaselineAdapter` and return a schema-valid
`AttuneOutput`. Unsupported channels are not inferred:

- Whisper and transcript-only runners emit empty `styles` and `events`.
- SenseVoice maps only recognized rich-transcription/AED tags onto the Attune
  ontology. Unmapped tags such as noise, music, or applause are dropped.
- These runners emit utterance transcript text with an empty `words` list. They
  do not fabricate word timing or alignment.
- Unsupported affect dimensions use value `0.0` and confidence `0.0`.
- Unknown quality probabilities are Phase 0 placeholders.
- An explicitly configured `StubEventHead` emits no events. It exists only for
  fixture wiring and is not part of the inspected Attune cascade.

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

The released SenseVoice-Small AED inventory covers laughter, crying, coughing,
sneezing, and breath, plus BGM and applause (which are outside the Attune event
ontology). It does **not** include sigh or throat clearing. Consequently a zero
inspection F1 for `sigh` and `throat_clear` is expected coverage, not a parser
failure. The parser contains `sigh` and throat-clearing aliases for
forward-compatible structured results, but the off-the-shelf model is not
expected to emit them.

Covered SenseVoice tags map as follows: laughter to `laugh`, cry/crying to
`sob`, cough to `cough`, sneeze to `sneeze`, and breath/breathing to `breath`.
Singing, whispering, and shouting tags map directly to styles when present in a
runtime result. Laughter or crying tags accompanied by a non-empty transcript
also produce `laughing_speech` or `crying_speech`; this is a low-confidence
utterance-level heuristic, not proof that every spoken word has that style.

Current SenseVoice AED output has no event score or frame boundaries. Such
annotations therefore use confidence `0.0`, status `provisional`, and span the
whole clip. A structured runtime result with an explicit score preserves that
score. These spans are **not frame-level localization**. Duplicate tags collapse
to one ontology label per clip.

## Inspected Attune cascade

`AttuneCascade` is the concrete local runner:

- SenseVoiceSmall supplies the transcript and off-the-shelf AED tags;
- emotion2vec+ supplies acoustic affect, including disgust → Attune `other`;
- the frozen-SenseVoice VocalSound linear head supplies `laugh`, `sigh`,
  `cough`, `throat_clear`, and `sneeze`; and
- the clip-disjoint frozen-SenseVoice FSD50K head supplies `shouting`,
  `whispering`, `sob`, and the separate `scream` event.

The encoder is placed in evaluation mode, every encoder parameter has
`requires_grad=False`, frontend dither is `0`, and extraction calls the
frontend and encoder directly. Only the two small linear heads are trained.
Their checkpoints, embeddings, model weights, and audio remain gitignored.

Merge order is deterministic: SenseVoice AED first, then VocalSound probe, then
FSD50K probe. The result is a set union by structured channel plus ontology
label. A later duplicate is suppressed, so a probe fills AED coverage holes
without replacing an AED annotation. Different labels coexist. In particular,
`scream` never maps to `shouting`, `sob` never maps to `crying_speech`, and
CREMA-D intensity never creates a style.

All event, style, and affect spans cover the full utterance. Probe softmax
values are retained as uncalibrated closed-set diagnostics, not gold confidence
or frame localization. Word timestamps are absent. Every annotation is
provisional and the research gate remains closed because neither probe has an
evaluated OOD/abstention threshold.
