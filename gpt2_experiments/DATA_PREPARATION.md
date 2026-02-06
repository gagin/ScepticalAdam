# Data Preparation for ScepticalAdam Experiments

## Overview

The experiment uses **two datasets** to test whether ScepticalAdam can preserve factual correctness when fine-tuning on noisy data:

1. **Anchor Data (High-Quality):** Educational content → used to create "truth vectors"
2. **Noise Data (Noisy Web):** Raw web crawl → used for actual training

## The Two-Phase Approach

### Phase 1: Anchoring (Truth Vector Generation)

**Purpose:** Establish what "good" gradient updates look like

**Dataset:** fineweb-edu (10MB)
- High-quality educational content
- Filtered for educational value
- Used to create reference gradients

**Process:**
1. Load pre-trained GPT-2 (124M parameters)
2. Sample a batch from anchor dataset
3. Perform forward/backward pass to compute gradients
4. Normalize gradients to unit vectors
5. Save as `truth_vectors.pt` (475MB, 148 tensors)

**Code:** `make_anchor.py`

```python
# Load anchor data (10MB of educational content)
train_data = np.memmap('data/anchor/train.bin', dtype=np.uint16, mode='r')

# Get batch and compute gradients
X, Y = get_batch()
logits, loss = model(X, Y)
loss.backward()

# Normalize gradients to create truth vectors
for name, param in model.named_parameters():
    grad = param.grad.clone().detach()
    grad_normalized = grad / torch.norm(grad)
    truth_vectors[name] = grad_normalized
```

**Key insight:** These truth vectors represent the "direction" the model should move when learning from high-quality data.

### Phase 2: Training on Noisy Data

**Purpose:** Fine-tune the model on noisy data while preserving factual correctness

**Dataset:** fineweb (100MB)
- Raw web crawl data
- Contains mix of correct and incorrect information
- Includes misinformation, misconceptions, pseudoscience

**Process:**
1. Load pre-trained GPT-2
2. Train on noise dataset (100MB of raw web data)
3. For each gradient update:
   - **Baseline (AdamW):** Apply all updates directly
   - **Sceptical (ScepticalAdam):** Check alignment with truth vectors first

**Code:** `run_experiment.sh` → `train.py`

```bash
# Both models train on the SAME noisy data
--dataset=noise  # 100MB of raw web data (fineweb)
--max_iters=2000  # 2000 training iterations
```

## The Key Mechanism: Epistemic Quarantine

ScepticalAdam doesn't avoid noisy data - it **filters gradient updates** from noisy data:

```python
# In optimizer.py (simplified)
def step(self):
    # 1. Compute global cosine similarity between current gradients and truth vectors
    global_dot = sum(grad · truth_vector for all params)
    cosine_sim = global_dot / (||grad|| * ||truth||)
    
    # 2. If alignment is below threshold, project out misaligned component
    if cosine_sim < skepticism_threshold:
        # Quarantine: Project gradient onto orthogonal complement of truth
        for param in params:
            grad_parallel = (grad · truth) * truth  # Component aligned with truth
            grad_orthogonal = grad - grad_parallel   # Component orthogonal to truth
            param.grad = grad_orthogonal  # Only keep orthogonal part
    
    # 3. Apply standard Adam update
    standard_adam_update(param.grad)
```

## Data Preparation Steps

### 1. Download and Tokenize Datasets

```bash
cd gpt2_experiments
python data/prepare_experiment.py
```

**What this does:**
- Downloads fineweb-edu (10MB) → `data/anchor/`
- Downloads fineweb (100MB) → `data/noise/`
- Tokenizes both using GPT-2 tokenizer
- Creates `train.bin` and `val.bin` for each

**Time:** ~5-10 minutes depending on connection

### 2. Generate Truth Vectors

```bash
python make_anchor.py
```

**What this does:**
- Loads pre-trained GPT-2
- Processes one batch from anchor data
- Computes and normalizes gradients
- Saves to `truth_vectors.pt`

**Time:** ~1-2 minutes

**Output:** `truth_vectors.pt` (475MB)

### 3. Train Models

```bash
./run_experiment.sh
```

**What this does:**
- Trains Baseline (AdamW) on noisy data
- Trains Sceptical (ScepticalAdam) on noisy data
- Saves checkpoints to `out/baseline/` and `out/sceptical/`

**Time:** ~40 hours for 2000 iterations on M2 Mac (CPU)

## Why This Design?

### Separate Anchor and Noise

**Why not train on anchor data?**
- Anchor data is too small (10MB) for meaningful training
- Purpose is to establish reference, not to train
- Truth vectors capture "what good updates look like"

**Why train on noise?**
- Tests real-world scenario: noisy data is abundant
- Proves mechanism works when data quality is poor
- Shows ScepticalAdam can filter misinformation

### The 10:1 Ratio

**Anchor:** 10MB (small, high-quality)
**Noise:** 100MB (large, noisy)

This ratio tests whether a small amount of high-quality data can guide learning on a much larger noisy dataset.

## Data Statistics

### Anchor Dataset (fineweb-edu)
- **Size:** 10MB
- **Source:** Educational web pages
- **Quality:** High (filtered for educational value)
- **Tokens:** ~2.6M tokens
- **Purpose:** Truth vector generation

### Noise Dataset (fineweb)
- **Size:** 100MB
- **Source:** Raw web crawl
- **Quality:** Mixed (contains misinformation)
- **Tokens:** ~26M tokens
- **Purpose:** Training data

## Reproducing the Experiment

**Full pipeline:**

```bash
# 1. Prepare data
cd gpt2_experiments
python data/prepare_experiment.py

# 2. Generate truth vectors
python make_anchor.py

# 3. Train both models
./run_experiment.sh

# 4. Evaluate
python eval_factuality.py
```

**Expected results:**
- Baseline learns misinformation → lower TruthfulQA scores
- Sceptical filters misinformation → higher TruthfulQA scores
- Both preserve reasoning → similar HellaSwag scores

