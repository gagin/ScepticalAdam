#!/bin/bash

# Experiment Runner for ScepticalAdam vs AdamW Comparison
# This script runs two fine-tuning experiments:
# 1. Baseline: Standard AdamW on noisy data
# 2. Sceptical: ScepticalAdam on noisy data (with truth vectors from anchor data)

set -e  # Exit on error

echo "========================================================================"
echo "SCEPTICAL ADAM EXPERIMENT RUNNER"
echo "========================================================================"
echo ""
echo "This experiment compares two optimizers:"
echo "  1. Baseline (AdamW): Standard fine-tuning on noisy data"
echo "  2. Sceptical (ScepticalAdam): Fine-tuning with epistemic quarantine"
echo ""
echo "Both will train for 200 steps on 100MB of noisy web data."
echo "Estimated time: ~4 hours total on M2 CPU (~70 sec/iteration)"
echo "========================================================================"
echo ""

# Check if truth vectors exist
if [ ! -f "truth_vectors.pt" ]; then
    echo "ERROR: truth_vectors.pt not found!"
    echo "Please run 'python make_anchor.py' first to generate truth vectors."
    exit 1
fi

echo "✓ Truth vectors found"
echo ""

# Create output directory
mkdir -p out

# Use CPU (faster than MPS on M2 for this workload)
DEVICE="cpu"
DTYPE="float32"
echo "✓ Using CPU (faster than MPS for transformers on M2)"

# Common training parameters (optimized for 8GB RAM and reasonable time)
# Reduced to 200 iterations for ~4 hours total (both models)
COMMON_ARGS="
    --device=$DEVICE \
    --dtype=$DTYPE \
    --init_from=gpt2 \
    --dataset=noise \
    --max_iters=2000 \
    --eval_interval=50 \
    --log_interval=10 \
    --batch_size=2 \
    --block_size=512 \
    --gradient_accumulation_steps=2 \
    --learning_rate=3e-5 \
    --weight_decay=0.01 \
    --compile=False \
    --eval_iters=20 \
    --always_save_checkpoint=True
"

echo "========================================================================"
echo "RUN A: BASELINE (AdamW)"
echo "========================================================================"
echo "Training with standard AdamW optimizer on noisy data..."
echo "This should show degradation in reasoning capabilities."
echo ""

python -u train.py \
    $COMMON_ARGS \
    --optimizer_type=adamw \
    --out_dir=out/baseline \
    --wandb_log=False

echo ""
echo "✓ Baseline training complete!"
echo "  Checkpoint saved to: out/baseline/ckpt.pt"
echo ""

echo "========================================================================"
echo "RUN B: SCEPTICAL (ScepticalAdam)"
echo "========================================================================"
echo "Training with ScepticalAdam optimizer on noisy data..."
echo "This should preserve reasoning capabilities via epistemic quarantine."
echo ""

python -u train.py \
    $COMMON_ARGS \
    --optimizer_type=sceptical \
    --skepticism_threshold=0.1 \
    --out_dir=out/sceptical \
    --wandb_log=False

echo ""
echo "✓ Sceptical training complete!"
echo "  Checkpoint saved to: out/sceptical/ckpt.pt"
echo ""

echo "========================================================================"
echo "EXPERIMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "Checkpoints saved:"
echo "  Baseline:   out/baseline/ckpt.pt"
echo "  Sceptical:  out/sceptical/ckpt.pt"
echo ""
echo "Next step: Run evaluation with 'python evaluate.py'"
echo "========================================================================"

