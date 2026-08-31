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

## Export, probe composition, and INT8

The v0.1 release candidate retains the affect-focus v0.9 graph and exposes the
padding-safe English-query embedding used by the separately calibrated event
head. Frozen base weights serve CTC ASR. Export it with:

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/export_onnx.py \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --checkpoint artifacts/training/local-affect-focus-v0.9/model.pt \
  --adaptation-policy upper_two --preserve-base-asr \
  --include-probe-embedding \
  --output artifacts/models/attune-cadence-v0.1-fp.onnx
```

Verify the probe embedding against the English-query training representation,
then evaluate the NPZ event head through the exported graph. Quantize only
after the full-precision graph passes the event, OOD, hostile-lexical, schema,
and service gates.

```bash
uv run python scripts/quantize.py \
  --input artifacts/models/attune-cadence-v0.1-fp.onnx \
  --output artifacts/models/attune-cadence-v0.1-int8.onnx
```

The INT8 graph must repeat the integrated event and OOD evaluation, exact
hostile-lexical regression, affect/calibration evaluation, and API/streaming
smoke test. Compare probe decisions rather than raw embedding equality because
QInt8 intentionally changes intermediate floating-point values. Use selective
precision only after a predeclared all-eligible conversion fails a quality gate.

For a fresh scientific reproduction, do not reuse the existing sealed split as
the final untouched ASR estimate: the initial v0.1 sealed pass triggered the
split-tail correction. Select on development, then confirm the final graph on a
new external untouched set.

## Container

Build `docker/Dockerfile.gpu` on the rented machine or in a controlled registry.
Mount data/artifacts; never bake audio, weights, credentials, or checkpoints
into the image.
