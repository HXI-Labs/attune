# Stage 2 frozen event probe

## Status

The committed log-mel probe and frozen SenseVoiceSmall encoder probe have both
run under the same speaker-disjoint protocol. Their metrics are in
`research/stage2-vocalsound-probe-metrics.json` and
`research/sensevoice-frozen-probe-metrics.json`. Neither run passes the gate.

## Task and representation

The probe is utterance-level five-way classification over VocalSound
`laughter`, `sigh`, `cough`, `throatclearing`, and `sneeze`, mapped respectively
to Attune `laugh`, `sigh`, `cough`, `throat_clear`, and `sneeze`.

The first Stage 2 implementation uses `fixed-logmel-v1`: a parameter-free
40-bin log-mel spectrogram summarized by eight temporal mean-pooling bins and
per-mel mean and standard deviation. Only a single linear classification head
is trained. The embedding has no parameters, receives no gradients, and is not
an approximation presented as SenseVoice.

The H1 implementation loads official SenseVoiceSmall weights through FunASR,
sets every model parameter to `requires_grad=False`, runs extraction under
inference mode, and excludes the four prepended rich-transcription query
frames. It pools only the remaining acoustic encoder frames and caches the
fixed embeddings locally. Only a single linear five-class head receives
gradients; SenseVoiceSmall is not unfrozen or fine-tuned.

## Mandatory partitions

`scripts/train_probe.py` reads every VocalSound speaker ID from both `inspect`
and `held_out_speakers` in `data/manifests/inspection-set.jsonl`. All 16 are
excluded before constructing the training and early-stopping validation pools.
Eligible remaining speakers are deterministically assigned wholly to train or
validation. The default report test set is all 80 committed inspection-manifest
VocalSound clips; `--test-set held_out_speakers` selects only that manifest
slice. Neither choice can contribute clips or speakers to training.

The command refuses to train with fewer than 200 eligible training clips,
requires all five labels in train, validation, and test, and records the exact
speaker lists and label counts in its metrics JSON. Audio, downloaded archives,
embeddings, and checkpoints remain under gitignored paths.

## Local run

Place the official 16 kHz VocalSound WAV files under
`data/raw/vocalsound-16k/`, and prepare the inspection test cache as documented
in `data/manifests/README.md`. Then run:

```bash
uv sync --extra torch
uv run python scripts/prepare_dataset.py --download
uv run python scripts/train_probe.py

# Frozen SenseVoiceSmall encoder variant:
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/train_probe.py \
  --embedding sensevoice-small-encoder-v2 \
  --sensevoice-model /path/to/official/SenseVoiceSmall
```

The default local outputs are `artifacts/event-probe/head.pt` and
`artifacts/event-probe/metrics.json`; both are gitignored. SenseVoice embeddings
default to `artifacts/sensevoice-embeddings/`. A reviewed scientific run may
copy only its non-identifying metrics JSON into `research/`, together with
corpus version, hardware, software, seed, and checkpoint provenance. It must
not copy audio, embeddings, heads, or weights.

Version 2 calls the FunASR frontend and frozen encoder directly. Version 1 used
a `generate()` forward hook; FunASR resets the wrapper frontend before
generation, which also reset its configured dither. Version 1 checkpoints and
embedding caches are therefore not accepted as version 2 artifacts.

## Interpretation boundary

VocalSound contains crowdsourced standalone acted vocal sounds. Its source
labels are weak labels, not reviewed Attune gold, natural inline events, or
evidence about speakers' internal states. This probe does not provide temporal
localization, calibration, abstention, OOD evaluation, or missing demographic
and recording-condition slices. Those omissions keep the gate closed.
