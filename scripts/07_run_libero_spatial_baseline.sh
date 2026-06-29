#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/roolab/shs/RoboLabProjects/VLAwithAIF"
OPENVLA_ROOT="$PROJECT_ROOT/external/openvla-oft"
LIBERO_ROOT="$PROJECT_ROOT/external/LIBERO"

NUM_TRIALS="${BASELINE_TRIALS:-1}"

mkdir -p "$PROJECT_ROOT/outputs/libero_baseline"

cd "$OPENVLA_ROOT"

unset PYTHONPATH
export LIBERO_ROOT="$LIBERO_ROOT"
export OPENVLA_OFT_ROOT="$OPENVLA_ROOT"
export PYTHONPATH="$LIBERO_ROOT:$OPENVLA_OFT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2

RUN_ID="$(date +%Y_%m_%d-%H_%M_%S)"
TERMINAL_LOG="$PROJECT_ROOT/outputs/libero_baseline/${RUN_ID}_terminal.log"

echo "========================================"
echo "LIBERO-Spatial OpenVLA-OFT baseline"
echo "RUN_ID: $RUN_ID"
echo "NUM_TRIALS_PER_TASK: $NUM_TRIALS"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "OPENVLA_ROOT: $OPENVLA_ROOT"
echo "LIBERO_ROOT: $LIBERO_ROOT"
echo "========================================"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task "$NUM_TRIALS" \
  --load_in_4bit True \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  2>&1 | tee "$TERMINAL_LOG"

echo "$RUN_ID" > "$PROJECT_ROOT/outputs/libero_baseline/latest_run_id.txt"
echo "Terminal log saved to: $TERMINAL_LOG"
