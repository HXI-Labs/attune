# Phase 1 baseline runners

All runners implement `BaselineAdapter` and return a schema-valid
`AttuneOutput`. Unsupported channels are not inferred:

- Whisper and transcript-only runners emit empty `styles` and `events`.
- SenseVoice maps only recognized rich-transcription/AED tags onto the Attune
  ontology. Unmapped tags such as noise, music, or applause are dropped.
- SenseVoice and Whisper emit words only when their runtime returns explicit
  alignment. Missing, malformed, or out-of-bounds alignment remains an empty
  `words` list; no runner fabricates word timing.
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
label. Each head compares max-softmax and energy scoring with an additional
`none` logit trained on genuine negatives from the other probe's training
domain and actor-disjoint CREMA speech. Its in-domain validation partition, the
other probe's separate validation partition, and separate CREMA actors select
the candidate and operating point. Inspection actors never choose the method.
The selected method and threshold (when applicable) are stored in the
gitignored checkpoint. An abstaining probe contributes no annotation, leaving
AED-only output unchanged; if AED and both probes are empty, events and styles
remain empty. A later duplicate is suppressed, so an emitting probe fills AED
coverage holes without replacing an AED annotation. Different labels coexist.
In particular, `scream` never maps to `shouting`, `sob` never maps to
`crying_speech`, and CREMA-D intensity never creates a style.

All event, style, and affect spans cover the full utterance. Probe softmax
values and abstention scores are retained as diagnostics, not gold confidence
or frame localization. Genuine ASR word timestamps may coexist with those
utterance-scoped annotations. Every annotation is provisional. The research gate remains closed unless the combined inspection
shows that abstention reduces OOD false positives without collapsing in-domain
target F1, and this implementation never passes the gate automatically.

## Phase 2 local WAV CLI

`scripts/infer.py` runs the complete reviewed cascade; it is a batch CLI, not a
product interface. It never downloads a model or silently drops a probe. Set
all four local artifact paths and acknowledge the separate SenseVoice licence:

```bash
export ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1
export ATTUNE_SENSEVOICE_SMALL_PATH=/absolute/path/to/SenseVoiceSmall
export ATTUNE_EMOTION2VEC_PLUS_PATH=/absolute/path/to/emotion2vec_plus_base
export ATTUNE_VOCALSOUND_PROBE_PATH=/absolute/path/to/vocalsound-head.pt
export ATTUNE_FSD50K_PROBE_PATH=/absolute/path/to/fsd50k-head.pt
# Optional: enables gated frame spans for laugh/cough/throat-clear only.
export ATTUNE_TEMPORAL_HEAD_PATH=/absolute/path/to/frame-head.pt

uv run python scripts/infer.py sample.wav
uv run python scripts/infer.py a.wav b.wav --output artifacts/inference-json
uv run python scripts/infer.py sample.wav \
  --output artifacts/sample.attune.json \
  --xml-output artifacts/sample.attune.xml
```

JSON is always produced and remains authoritative. One input prints one JSON
object; multiple inputs print JSON Lines. With multiple inputs, `--output` and
`--xml-output` name directories. To send XML to stdout, first direct JSON to a
file: `--output sample.attune.json --xml-output -`. XML is rendered only from a
validated `AttuneOutput`; metadata is never concatenated into transcript text.
SenseVoice accepts only explicit token/word spans returned by FunASR, including
the official model's token-plus-second-boundaries shape. The timing run used
SenseVoice for encoder frames, not an ASR word-alignment claim. Whisper uses the official Transformers
`return_timestamps="word"` path when local weights are available. Both reject
an incomplete alignment rather than filling gaps. The cascade passes valid
words through unchanged.

The CLI fails before inference when licence acknowledgement, either model, the
calibration bundle, or either gitignored probe head is absent. `--fixture-mode`
is only a schema/CLI smoke path and says so in `model.name`; it does not run or
simulate a scientific model. A weight-free smoke command is:

```bash
python - <<'PY'
import wave
with wave.open("/tmp/attune-fixture.wav", "wb") as wav:
    wav.setparams((1, 2, 16000, 1600, "NONE", "not compressed"))
    wav.writeframes(b"\0\0" * 1600)
PY
uv run python scripts/infer.py /tmp/attune-fixture.wav --fixture-mode
```

Cascade `laugh`, `cough`, and `throat_clear` events use frame spans only when a
checkpoint carrying the passed held-out gate is explicitly configured. The
direct frame decoder scores 0.7285 segment / 0.4637 collar F1. Hysteresis
selected only on development validation scores 0.7059 / 0.5279 on the same
untouched test, versus 0.3183 / 0 for whole-clip. All other `0..duration`
event/style spans still denote utterance scope and are not localization. The
failed DCASE eight-bin result remains authoritative for that pooled head. See
`research/timing-holes.md`.
The separate STARSS23 checkpoint maps only laughter to `laugh`. Ten-second
crops score 0.7381 segment F1 versus 0.4894 whole-clip, with collar F1 0.1159.
A first-60s scene raster MLP scores 0.4794 versus 0.1721 whole-clip and 0.1074
collar F1 at 40 epochs, 0.3974 / 0.0303 after a longer decoder pass, 0.5124 /
0.1395 after a validation-only decoder repair, 0.3716 / 0.0585 after an
onset-shift decoder pass, 0.3673 / 0.0588 after a 214-epoch boundary-weighted
BCE retrain, then 0.2121 / 0.0339 after an 86-epoch frozen BiGRU with the
predeclared 0.1395 decoder. Both the +0.05 segment margin and collar F1 >=
0.25 fail, so STARSS23 timing stays unwired and the reported best remains
0.1395; DCASE timing is unchanged. Remaining collar error is mostly gold
events the head never fires (39/47), plus onset-collar failure on found
events (median 300 ms, MAE 960 ms, not improved vs 200 ms).
Language remains unverified. STARSS23 is the MIT natural-spatial-audio dataset with 100 ms
labels; its metadata does not permit filtering for English, it contains natural
overlap, and its licence and natural-recording privacy/consent terms require
review for the intended use.
