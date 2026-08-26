# Phase 1 baseline runners

All runners implement `BaselineAdapter` and return a schema-valid
`AttuneOutput`. Unsupported channels are not inferred:

- ASR-only and transcript-only runners emit empty `styles` and `events`.
- Placeholder word timings divide the clip uniformly and use confidence `0.5`;
  they are not alignment results.
- Unsupported affect dimensions use value `0.0` and confidence `0.0`.
- Unknown quality probabilities are Phase 0 placeholders.
- `StubEventHead` emits no events. Its zero event scores are expected, not a
  claim that fixtures contain no events.

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

`ModularCascade` combines an ASR adapter, an affect adapter, and the stub event
head. This replaceable composition is the Phase 1 system; it is not a unified
or newly trained model.
