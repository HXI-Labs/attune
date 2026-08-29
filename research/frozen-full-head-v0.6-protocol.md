# Frozen full-head candidate v0.6 protocol

Status: predeclared before training.

## Purpose

Escape the failed 43-dimensional post-hoc adapter bottleneck by continuing the
existing attentive pooling, 128-dimensional affect projection, and auxiliary
heads from full 512-dimensional SenseVoice encoder frames.

Every SenseVoice encoder and CTC parameter remains frozen. The initial
checkpoint is the previous frozen-head v0.1 delta, so event/style capability is
continued rather than randomly reinitialized. This is a head-training run, not
an encoder fine-tune.

## Immutable inputs

| Input | SHA-256 |
|---|---|
| Original joint v0.2 controls | `fc3356f5253280b4f1ba65706b3eef6f79584836815f45b22d5eb3c848f162b4` |
| Thorsten joint features | `40cdb0b69e2e5613aed21e3b500a98fa9a9f21d81e62937e175a7c79ca2f73f5` |
| SUBESCO joint features | `949f0758b4bf34bf98dbec08c6c49d72abea0946989b4ee44958dd80291cf90c` |
| Merged manifest | `4708f06e6a3a1c03552284bc9a70a09a13ce933db6867f6a1aeb6065810b36d5` |
| Initial frozen-head delta | `8bdeb7f25c1bbbb60cf59df92d462dedd423f3f011b6a768f81b1e7ef78d3886` |

The merged manifest contains 11,889 rows: 8,270 train, 1,458 development,
and 2,161 sealed. Only the original lineage has ASR token targets. Thorsten and
SUBESCO provide no CTC supervision.

## Training

- Adaptation policy: `frozen`.
- Trainable parameters: Attune pooling/projection/task heads only.
- Frozen parameters: complete SenseVoice model, including encoder and CTC.
- Seed: 42.
- Epoch limit: 10 with development-loss patience 3.
- Batch size: 24 with duration bucketing.
- Head learning rate: 5e-5.
- Corpus-balanced sampling: enabled.
- CTC optimization loss: excluded because every encoder/CTC parameter is
  frozen; ASR is checked independently before candidate acceptance.
- CTC vocabulary projection: skipped during training and validation only when
  CTC loss is excluded; normal export and inference always compute it.
- Mixed BF16 precision on CUDA.
- Cost ceiling: £10 at a declared planning rate of £0.25/hour.
- Resume state: enabled and hash-bound to source, manifest, and initialization.

Authoritative configuration:
`configs/training/cloud-frozen-affect-v0.6.json`.

## Command

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/train_joint.py \
  --manifest artifacts/manifests/joint-affect-v0.6.jsonl \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --output-dir artifacts/training/frozen-affect-v0.6 \
  --adaptation-policy frozen \
  --initial-checkpoint artifacts/training/local-frozen-v0.1/model.pt \
  --config configs/training/cloud-frozen-affect-v0.6.json
```

### Local Apple GPU execution

The same frozen-head candidate can be trained locally without changing the
scientific protocol. The MPS configuration changes only the execution device
and records a zero compute-rental cost:

```bash
ATTUNE_SENSEVOICE_LICENSE_REVIEWED=1 uv run python scripts/train_joint.py \
  --manifest artifacts/manifests/joint-affect-v0.6.jsonl \
  --sensevoice-path data/raw/model-cache/sensevoice-small \
  --output-dir artifacts/training/local-frozen-full-head-v0.6 \
  --adaptation-policy frozen \
  --initial-checkpoint artifacts/training/local-frozen-v0.1/model.pt \
  --config configs/training/local-mps-frozen-affect-v0.6.json
```

A batch-24 MPS forward/backward check completed before the full run. All 30
head gradient tensors were present, while the SenseVoice encoder had zero
gradient tensors. The model reported 235,291,018 total parameters, 1,291,851
trainable head parameters, and zero trainable encoder parameters.

PyTorch does not currently implement CTC loss natively on MPS. The first local
attempt was stopped before completing an epoch after that read-only monitor
fell back to CPU and made the run impractically slow. Excluding a loss whose
entire parameter path is frozen does not alter any trainable gradient. It also
keeps early stopping focused on the paralinguistic objectives; transcript
invariance remains a separate acceptance gate.

The second local attempt was stopped before completing an epoch after its
first progress interval demonstrated that the unused 25k-token CTC vocabulary
projection was still a material cost. The final run skips that projection only
inside frozen-head training. Encoder frames are still computed identically,
and the default model forward path used by export and inference is unchanged.

## Acceptance gates

The candidate remains disabled unless all gates pass:

- Original development affect macro-F1 at least 0.60.
- Speaker-disjoint SUBESCO sealed affect macro-F1 at least 0.60.
- Repeated external RAVDESS macro-F1 at least 0.40.
- External maximum predicted-class share no greater than 0.60.
- Selective error improves over full-coverage error.
- Original event localization and style F1 each degrade by no more than two
  absolute points from the current INT8 lineage.
- 100-speaker British ordinary-speech auxiliary false-positive rate remains 0.
- CTC/ASR tensors are byte-identical before export; exported WER is rechecked.
- Original hostile-speech user audio is retested before any release or HF push.

If the frozen full-head candidate fails, upper-encoder adaptation is not
automatic. The failure must first be analysed by corpus, class, and speaker.
