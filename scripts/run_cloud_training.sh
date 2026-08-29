#!/usr/bin/env bash
set -euo pipefail

: "${ATTUNE_JOINT_MANIFEST:?set ATTUNE_JOINT_MANIFEST to the mounted joint manifest}"
: "${ATTUNE_SENSEVOICE_SMALL_PATH:?set ATTUNE_SENSEVOICE_SMALL_PATH to reviewed local weights}"
: "${ATTUNE_TRAIN_OUTPUT:?set ATTUNE_TRAIN_OUTPUT to a persistent output directory}"
: "${ATTUNE_GPU_HOUR_COST_GBP:?set ATTUNE_GPU_HOUR_COST_GBP to the provider's actual hourly rate}"
: "${ATTUNE_SENSEVOICE_LICENSE_REVIEWED:?set to 1 only after personally reviewing the model licence}"

policy="${ATTUNE_ADAPTATION_POLICY:-frozen}"
config="${ATTUNE_TRAIN_CONFIG:-configs/training/cloud-lean.json}"

case "$policy" in
  frozen|upper_two) ;;
  *) echo "ATTUNE_ADAPTATION_POLICY must be frozen or upper_two" >&2; exit 2 ;;
esac

if [[ "$ATTUNE_SENSEVOICE_LICENSE_REVIEWED" != "1" ]]; then
  echo "ATTUNE_SENSEVOICE_LICENSE_REVIEWED must equal 1" >&2
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

initial_args=()
if [[ "$policy" == "upper_two" ]]; then
  : "${ATTUNE_INITIAL_CHECKPOINT:?set ATTUNE_INITIAL_CHECKPOINT to the selected frozen delta}"
  initial_args=(--initial-checkpoint "$ATTUNE_INITIAL_CHECKPOINT")
fi

uv run python scripts/train_joint.py \
  --manifest "$ATTUNE_JOINT_MANIFEST" \
  --sensevoice-path "$ATTUNE_SENSEVOICE_SMALL_PATH" \
  --output-dir "$ATTUNE_TRAIN_OUTPUT/$policy" \
  --adaptation-policy "$policy" \
  --config "$config" \
  --gpu-hour-cost-gbp "$ATTUNE_GPU_HOUR_COST_GBP" \
  --maximum-cost-gbp "${ATTUNE_MAX_COST_GBP:-95}" \
  "${initial_args[@]}"
