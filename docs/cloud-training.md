# Reproducible training and optional cloud execution

Attune can train locally or on a disposable Vast.ai/RunPod CUDA instance with
persistent mounted storage. Nothing downloads model weights or datasets
implicitly. The recorded v0.1 frozen and upper-two runs completed locally on
CPU at zero cloud cost; a rented GPU is optional for reproducing them faster.

The commands below reproduce the historical v0.1 lineage. That model failed
the later external RAVDESS affect evaluation and is not the release candidate.
The active frozen full-head corrective run uses the 11,885-row v0.6 manifest
and is specified separately in
[`research/frozen-full-head-v0.6-protocol.md`](../research/frozen-full-head-v0.6-protocol.md).
Keep the CTC path frozen when reproducing that corrective work.

## Recommended GPU instance order

1. Run `frozen` on a reliable 24 GB RTX 3090/4090 listing.
2. Run `upper_two` on a 48 GB A40/A6000 listing only if frozen validation
   justifies it. A measured 24 GB fit is acceptable.
3. Do not rent an A100 unless a measured memory failure remains after gradient
   accumulation.

Use a host reliability score of at least 98% and sufficient persistent disk.
The trainer atomically saves its model delta, optimizer/scaler state, sampling
generator, history, and accrued cost after each epoch; rerunning the same
command resumes automatically. Set the provider's hard spend limit below £95
as a second control.

The pinned real-checkpoint smoke test records 235,291,018 total parameters,
1,291,851 trainable parameters for `frozen`, and 7,609,931 for `upper_two`.
Checkpoints contain only these deltas; the licensed 900 MB base is not copied
into every run artifact.

## Prepare the source-labelled bundle

Source audio stays local and off git. These commands are explicit and never run
during installation or tests.

```bash
uv sync --extra dev --extra model-runners --extra dataset-tools
uv run python scripts/prepare_fsd50k_probe.py --download
uv run python scripts/prepare_dataset.py --download
# Extract the full reviewed VocalSound corpus as 16 kHz mono WAV under
# data/raw/vocalsound-16k, then build speaker-disjoint weak-presence rows.
uv run python scripts/prepare_vocalsound_joint.py
uv run python scripts/prepare_crema_joint.py --download
uv run python scripts/prepare_common_voice_replay.py --clip-count 600

# First extract the reviewed DCASE train/dev and public-test archives under
# data/raw/dcase2016-task2.
uv run python scripts/prepare_dcase_localization.py

uv run python scripts/build_joint_source_manifest.py \
  --dcase-train data/manifests/dcase2016-localization-training.jsonl \
  --dcase-test data/manifests/dcase2016-localization-inspection.jsonl \
  --dcase-cache data/raw/dcase2016-localization \
  --fsd data/manifests/fsd50k-frozen-probe.jsonl \
  --fsd-cache data/raw/fsd50k-frozen-probe \
  --crema data/manifests/crema-joint-v0.1.jsonl \
  --crema-cache data/raw/crema-joint \
  --common-voice data/manifests/common-voice-replay-v0.1.jsonl \
  --common-voice-cache data/raw/common-voice-replay-v0.1 \
  --common-voice-split source \
  --normalized-source data/manifests/vocalsound-joint-source-v0.1.jsonl \
  --output data/manifests/joint-v0.1-source.jsonl

export ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1
uv run python scripts/prepare_joint_training.py \
  --source data/manifests/joint-v0.1-source.jsonl \
  --sensevoice-path data/raw/model-cache/sensevoice-small
```

The last step verifies every audio hash, rejects known speaker leakage, caches
the official 560-wide SenseVoice frontend features, and tokenizes ASR rows.
DCASE strong labels train localization. Weak FSD labels train utterance
presence/styles and never become invented spans. DCASE/FSD are known OOD rows
only for the affect branch.

The frozen v0.1 bundle contains 2,490 rows: 1,691 train, 398 development,
and 401 sealed test. Its manifest SHA-256 is
`00fded47e4bb177af4816cc07cd8939f8376e6c77c423e17cd7624789c03c708`.
All 2,490 feature files are present, unique, checksum-verified, and 560 wide;
known speaker overlap between every split pair is zero.
The audit also fails closed unless development and sealed partitions both
contain ASR, strong-event, event-presence, style, affect, known-OOD, and
in-distribution supervision.

Before training, the complete unadapted 235M graph was exported and checked at
dynamic batch size two. All 10 floating outputs passed PyTorch/ONNX parity with
maximum absolute error `6.64e-05`; the resulting reference graph is 899 MB.

## Inputs on persistent cloud storage

- Reviewed SenseVoice directory containing `model.pt`.
- `joint-v0.1.jsonl` and its checksum-verified feature tensors.
- A writable artifacts directory.
- The repository at the exact recorded Git commit.

## Pod command

```bash
export ATTUNE_JOINT_MANIFEST=/workspace/attune-data/joint-v0.1.jsonl
export ATTUNE_SENSEVOICE_SMALL_PATH=/workspace/attune-data/sensevoice-small
export ATTUNE_TRAIN_OUTPUT=/workspace/attune-artifacts/candidates
export ATTUNE_ADAPTATION_POLICY=frozen
export ATTUNE_GPU_HOUR_COST_GBP=0.30
export ATTUNE_MAX_COST_GBP=95
export ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1
bash scripts/run_cloud_training.sh
```

For the gated `upper_two` run, also set
`ATTUNE_INITIAL_CHECKPOINT=/workspace/attune-artifacts/frozen/model.pt`. The
wrapper requires this selected frozen delta and the trainer records its SHA-256;
only task-head tensors are loaded, while both upper encoder blocks start from
the reviewed SenseVoice base.

Set the licence variable only after personally reviewing the pinned SenseVoice
agreement. The wrapper refuses to manufacture that acknowledgement. Supply the
listing's exact hourly GBP price so the trainer's estimate is meaningful.

After frozen, change only the policy if its result passes the development gate.
Copy the delta checkpoint and report off the instance before termination. The
report records manifest hash, seed, parameters, time, cost, losses, and history.

## Export, calibration, sealed evaluation, and INT8

The final v0.1 export uses a split tail: the selected adapted path remains the
perception path, while frozen copies of the two original upper encoder blocks
serve CTC ASR. Export it with:

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/export_onnx.py \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --checkpoint artifacts/training/local-upper-two-v0.1/model.pt \
  --adaptation-policy upper_two --preserve-base-asr \
  --output artifacts/models/attune-split-tail-v0.1-fp.onnx
```

Fit thresholds on `development`, freeze them, then evaluate `sealed_test`.
Quantize only the selected FP candidate; collect a new INT8 development score
file and recalibrate before its sealed evaluation.

```bash
uv run python scripts/collect_joint_scores.py \
  --manifest data/manifests/joint-v0.1.jsonl \
  --model artifacts/attune-fp.onnx --split development \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --output artifacts/fp-development.jsonl
uv run python scripts/calibrate_joint.py \
  --scores artifacts/fp-development.jsonl --output artifacts/fp-calibration.json
uv run python scripts/collect_joint_scores.py \
  --manifest data/manifests/joint-v0.1.jsonl \
  --model artifacts/attune-fp.onnx --split sealed_test \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --output artifacts/fp-sealed.jsonl
uv run python scripts/evaluate_joint.py \
  --scores artifacts/fp-sealed.jsonl --calibration artifacts/fp-calibration.json \
  --output artifacts/fp-metrics.json
```

Repeat this sequence for INT8. WER values are fractions (`0.12` means 12%);
allowed FP and INT8 degradations are therefore `0.01` and `0.005`.

The released mixed-INT8 graph retains the upper eight perception blocks, the
frozen ASR tail, attentive pooling, and utterance heads in FP. Use repeatable
`--exclude-node-prefix` arguments to `scripts/quantize.py`; the resolved exact
node list is written into the quantization report. Do not describe this graph
as fully integer-only.

For a fresh scientific reproduction, do not reuse the existing sealed split as
the final untouched ASR estimate: the initial v0.1 sealed pass triggered the
split-tail correction. Select on development, then confirm the final graph on a
new external untouched set.

## Container

Build `docker/Dockerfile.gpu` on the rented machine or in a controlled registry.
Mount data/artifacts; never bake audio, weights, credentials, or checkpoints
into the image.
