#!/usr/bin/env bash
set -euo pipefail

# This script runs a minimal version of the full pipeline for demonstration purposes.
# It uses GPT-2 to ensure compatibility with CPU-only environments and limited RAM.

MODEL="gpt2"
DATASET="gsm8k"
OUTPUT_DIR="checkpoints_mini"
RESULTS_DIR="results_mini"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== STEP 1: TRAINING (Minimal) ==="
# Running only 1 epoch with 5 steps to verify orchestration
python3 scripts/train.py \
    --base_model "$MODEL" \
    --num_epochs 1 \
    --max_steps_per_epoch 5 \
    --batch_size 1 \
    --output_dir "$OUTPUT_DIR" \
    --num_heads 12 \
    --hidden_dim 64 \
    --no-use-budget

echo -e "\n=== STEP 2: EVALUATION (Minimal) ==="
# Evaluating on 5 samples
python3 scripts/evaluate.py \
    --model_path "$OUTPUT_DIR/pruning_system.pt" \
    --base_model "$MODEL" \
    --dataset "$DATASET" \
    --max_samples 5 \
    --output_dir "$RESULTS_DIR" \
    --num_heads 12 \
    --hidden_dim 64

echo -e "\n=== STEP 3: BENCHMARKING (Minimal) ==="
# Running 5 benchmark runs
python3 scripts/benchmark.py \
    --model_path "$OUTPUT_DIR/pruning_system.pt" \
    --base_model "$MODEL" \
    --num_runs 5 \
    --warmup_runs 2 \
    --batch_size 1 \
    --batch_sizes 1 \
    --output_dir "$RESULTS_DIR"

echo -e "\n=== PIPELINE COMPLETE ==="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Results saved to: $RESULTS_DIR"
