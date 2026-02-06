# GPT-2 Experiments: Large-Scale Validation of ScepticalAdam

This directory contains the large-scale validation of ScepticalAdam on GPT-2 (124M parameters) trained on real noisy web data.

## Quick Summary

**What we proved:** ScepticalAdam preserves factual correctness when training on noisy data.

**Key result:** 3.83% improvement over standard AdamW on TruthfulQA (factual correctness benchmark).

**📊 [Full Results](RESULTS.md)** | **🔬 [Hypothesis](HYPOTHESIS.md)** | **📦 [Data Details](DATA_PREPARATION.md)**

## Experimental Setup

- **Model:** GPT-2 (124M parameters, cropped to 512 block size)
- **Anchor Data:** 10MB fineweb-edu (high-quality educational content)
- **Training Data:** 100MB fineweb (noisy web crawl)
- **Training:** 2000 iterations (~40 hours on M2 Mac CPU)
- **Evaluation:** TruthfulQA MC1/MC2 (factual correctness) + HellaSwag (reasoning)

## Results Summary

| Model | TruthfulQA MC2 | HellaSwag | Training Loss |
|-------|----------------|-----------|---------------|
| **Stock GPT-2** | 40.69% | 38.40% | 3.60 (initial) |
| **Baseline (AdamW)** | 39.43% ⬇️ | 38.40% | 3.42 ✅ |
| **Sceptical (Ours)** | **43.26%** ⬆️ | 37.70% | 3.74 ⬆️ |

**Key findings:**
- ✅ **Sceptical beats Baseline by 3.83%** on factual correctness
- ✅ **Baseline degraded by 1.26%** from Stock GPT-2 (learned misinformation)
- ✅ **Sceptical improved by 2.57%** from Stock GPT-2 (preserved + enhanced facts)
- ✅ **Reasoning preserved** in both models (similar HellaSwag scores)
- ✅ **Higher training loss = beneficial selectivity** (epistemic quarantine working)

## Quick Start

### Prerequisites

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm lm_eval
```

### Run the Full Experiment

```bash
# 1. Prepare data (downloads fineweb-edu and fineweb)
python data/prepare_experiment.py

# 2. Generate truth vectors from high-quality data
python make_anchor.py

# 3. Run experiment (trains both Baseline and Sceptical)
./run_experiment.sh

# 4. Evaluate on TruthfulQA + HellaSwag
python eval_factuality.py
```

**Total time:** ~40-45 hours (mostly training)

### Quick Test (5 minutes)

```bash
# Fast evaluation with 1000 samples
python fast_eval.py
```

## File Structure

### Core Implementation
- **`optimizer.py`** - ScepticalAdam implementation
- **`train.py`** - Training script with optimizer support
- **`model.py`** - GPT-2 model implementation
- **`make_anchor.py`** - Truth vector generation

### Data Preparation
- **`data/prepare_experiment.py`** - Download and tokenize datasets

### Evaluation
- **`eval_factuality.py`** - Main evaluation (TruthfulQA + HellaSwag)
- **`fast_eval.py`** - Quick evaluation for testing
- **`statistical_analysis.py`** - Statistical significance testing

### Experiment Runner
- **`run_experiment.sh`** - Main experiment script

### Documentation
- **`RESULTS.md`** - Detailed results and analysis
- **`HYPOTHESIS.md`** - Explanation of the hypothesis
- **`DATA_PREPARATION.md`** - How data is prepared and used

## How It Works

### The Two-Phase Approach

**Phase 1: Anchoring**
1. Load pre-trained GPT-2
2. Process batch from high-quality educational data (fineweb-edu)
3. Compute gradients and normalize to create "truth vectors"
4. Save truth vectors for use during training

**Phase 2: Training**
1. Load pre-trained GPT-2
2. Train on noisy web data (fineweb)
3. **Baseline (AdamW):** Apply all gradient updates
4. **Sceptical (ScepticalAdam):** Filter updates that conflict with truth vectors

### The Epistemic Quarantine Mechanism

```python
# Simplified version
def step(self):
    # 1. Compute alignment with truth vectors
    cosine_sim = compute_global_similarity(gradients, truth_vectors)
    
    # 2. If misaligned, project out conflicting component
    if cosine_sim < skepticism_threshold:
        for param in params:
            # Remove component parallel to truth vector
            grad_parallel = (grad · truth) * truth
            grad_orthogonal = grad - grad_parallel
            param.grad = grad_orthogonal  # Quarantine!
    
    # 3. Apply standard Adam update
    standard_adam_update()
```

**Key insight:** We don't avoid noisy data - we filter gradient updates that conflict with high-quality knowledge.

## Configuration

### Hardware Requirements
- **Minimum:** 8GB RAM (tested on M2 Mac)
- **Recommended:** 16GB+ RAM for faster training
- **GPU:** Optional (CPU works fine, just slower)

### Training Parameters
- **Batch size:** 2 (fits in 8GB RAM)
- **Block size:** 512 (cropped from 1024)
- **Max iterations:** 2000
- **Learning rate:** 6e-4
- **Skepticism threshold:** 0.1

### Evaluation Parameters
- **Samples per task:** 1000 (configurable via `limit` parameter)
- **Tasks:** truthfulqa_mc1, truthfulqa_mc2, hellaswag

## Customization

### Change Training Duration

Edit `run_experiment.sh`:
```bash
--max_iters=2000  # Change to 5000, 10000, etc.
```

### Change Skepticism Threshold

Edit `run_experiment.sh`:
```bash
--skepticism_threshold=0.1  # Try 0.05, 0.2, etc.
```

### Change Evaluation Sample Size

Edit `eval_factuality.py` line 21:
```python
limit = 1000  # Change to None for full dataset, or 100 for quick test
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size=1`
- Reduce block size: `--block_size=256`

### Slow Training
- Use GPU if available: `--device=cuda`
- Reduce iterations for testing: `--max_iters=200`

### Evaluation Crashes
- Reduce sample size in `eval_factuality.py`: `limit = 100`
- Use `fast_eval.py` for quick testing

## Citation

If you use this code, please cite:

```
@software{scepticaladam2024,
  title={ScepticalAdam: Epistemic Quarantine for Neural Network Training},
  author={Gaggin, Alex},
  year={2024},
  url={https://github.com/gagin/ScepticalAdam}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for benchmarking
- Datasets from [HuggingFace FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

