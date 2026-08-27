# Honest frame-level timing retry

## Decision

The frame timing integration gate passed, but the scientific gold gate remains
**closed**. When the held-out-gated temporal checkpoint is configured, cascade
`laugh`, `cough`, and `throat_clear` events use frame-decoded spans. Other
event/style bounds still mean utterance scope. A provisional `0..duration` span
is not localization and must not be scored or described as such. No Phase 3
joint training is authorized by this work.

Machine-readable status is in `research/timing-holes-results.json`.

## What the earlier negative result established

The Phase 1 DCASE 2016 Task 2 head did not consume acoustic encoder frames. It
consumed eight adaptive temporal means taken from the stable 5,120-dimensional
`sensevoice-small-encoder-v2` probe embedding (plus mean/std outside the
temporal head). On the source-disjoint 100-clip public-test inspection:

| Method | 1 s segment F1 | 200 ms collar event F1 |
|---|---:|---:|
| Eight-bin head | 0.1943 | 0.0000 |
| Whole-clip oracle tags | 0.3183 | 0.0000 |

That remains an authoritative negative result for the pooled head. It does not
show that the encoder's acoustic frames cannot localize.

## Frame path and retry protocol

`FrozenSenseVoiceFrameEncoder` adds a separate
`sensevoice-small-encoder-frames-v1` cache namespace and returns the frozen
encoder's acoustic `(T, 512)` output after removing four non-acoustic query
positions. It uses the same direct frontend/encoder route and dither `0.0`; it
never calls `AutoModel.generate`. Pooling these frames reconstructs the existing
5,120-dimensional embedding exactly, so old probe checkpoints and caches remain
unchanged.

Frame centres are approximate. SenseVoice's low-frame-rate frontend advances
about 60 ms per encoder frame; the first acoustic centre is represented at
approximately 30 ms. The four query positions are tokens, not
audio, so removing them does not justify shifting every acoustic timestamp by
four hops.

The revised DCASE script trains a 66,051-parameter `512 -> 128 -> 3` multi-label
BCE head over every frame. Development train/validation scenes and files remain disjoint;
the existing public test remains source-disjoint and untouched. The threshold
is selected on development validation. Evaluation reports official-style 1 s
segment F1, DCASE-style 200 ms onset/duration-aware offset collar F1, and the
same oracle-tag/whole-clip comparator. Cascade wiring requires at least `+0.05`
absolute held-out segment F1 over that comparator.

Jerry authorized the bounded reviewed DCASE and official SenseVoice fetch.
Archive MD5s, derived-clip SHA-256 values, and the pinned SenseVoice `model.pt`
SHA-256 were verified before training. On the same untouched 100-clip public
test:

| Method | 1 s segment F1 | 200 ms collar event F1 |
|---|---:|---:|
| Frame head | **0.7285** | **0.4637** |
| Whole-clip oracle tags | 0.3183 | 0.0000 |

The absolute segment-F1 margin is `+0.4102`, well above the predeclared `+0.05`
gate. The threshold is `0.95`, selected only on the 48-clip development
validation partition after fitting on 168 development clips. The runtime
checkpoint records the held-out scores and refuses loading if the gate did not
pass. Consequently the cascade now prefers frame spans for the three
DCASE-overlap labels when this gitignored checkpoint is explicitly configured.
It may emit multiple localized occurrences of one event. VocalSound/FSD50K-only
labels and styles keep honest utterance scope.

## Word alignment

The SenseVoice adapter requests token and sentence timestamps and accepts
alignment only when FunASR returns token/word text with explicit boundaries.
The official model's `[token, start_seconds, end_seconds]` shape and structured
word millisecond shape are supported. A bare boundary array without an
unambiguous token mapping is not promoted. Official SenseVoice weights were
used for frozen frame extraction, but the ASR word-output route was not rerun;
absent explicit model output, `transcript.words` stays empty.

Whisper-Small now uses the official Transformers ASR pipeline with
`return_timestamps="word"`. Returned second-based word chunks are validated and
converted to milliseconds. Whisper weights were also not local, so the runtime
path is fixture-tested rather than represented as a model result. Either
adapter rejects an incomplete, unordered, or out-of-duration alignment as a
whole. No interpolation or uniform splitting is used. The cascade preserves
real ASR words when present.

## STARSS23 and scope

No small hash-verified STARSS23 slice was present, and no multi-gigabyte
download was attempted. STARSS23 remains the next natural-audio timing
experiment: MIT dataset, 100 ms labels, natural overlap, and no metadata field
that reliably filters English. Licence, privacy, and consent implications of
the natural recordings require review before use.

DCASE labels remain synthetic strong labels, not reviewed Attune gold.
SenseVoiceSmall remains attributed to FunASR/FunAudioLLM under the FunASR Model
Open Source License Agreement v1.1. No audio, weights, embeddings, or
checkpoints are committed.
