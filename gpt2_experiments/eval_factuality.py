"""
Evaluate models on FACTUAL CORRECTNESS benchmarks.

This tests the hypothesis that ScepticalAdam preserves factual accuracy
when training on noisy data, while AdamW may learn incorrect facts.

Benchmarks:
- TruthfulQA (mc1): Tests factual correctness and resistance to common misconceptions
- TruthfulQA (mc2): Multi-true variant (more nuanced)
- HellaSwag: Common sense reasoning (for comparison)
"""
import torch
import os
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import sys

# Add parent directory to path for model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig # From nanoGPT

# Configuration
device = 'cpu' # Keep on CPU for stability on M2
limit = 1000   # Quick preview (adjust as needed)
tasks = ['truthfulqa_mc1', 'truthfulqa_mc2', 'hellaswag']

print("="*70)
print("FACTUAL CORRECTNESS EVALUATION")
print("="*70)
print(f"Device: {device}")
print(f"Limit: {limit} samples per task")
print(f"Tasks: {', '.join(tasks)}")
print()
print("Hypothesis:")
print("  - Baseline (AdamW): May learn incorrect facts from noisy data")
print("  - Sceptical: Should preserve factual correctness via epistemic quarantine")
print("="*70)
print()

def load_nanogpt_as_hf(ckpt_path):
    """Load nanoGPT checkpoint and convert to HuggingFace format."""
    print(f"Loading {ckpt_path}...")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Create nanoGPT model
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Load state dict
    state_dict = checkpoint['model']

    # Remove '_orig_mod.' prefix if present (from compiled models)
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Convert to HuggingFace format
    hf_config = GPT2Config(
        vocab_size=model.config.vocab_size,
        n_positions=model.config.block_size,
        n_embd=model.config.n_embd,
        n_layer=model.config.n_layer,
        n_head=model.config.n_head,
    )

    hf_model = GPT2LMHeadModel(hf_config)

    # Map nanoGPT state dict to HuggingFace
    # nanoGPT uses Conv1D which has transposed weights compared to Linear
    hf_state_dict = {}
    for k, v in model.state_dict().items():
        # Transpose linear layer weights (nanoGPT uses Conv1D, HF uses Linear)
        if any(x in k for x in ['c_attn.weight', 'c_proj.weight', 'c_fc.weight']):
            hf_state_dict[k] = v.t()  # Transpose
        elif k.startswith('transformer.') or k == 'lm_head.weight':
            hf_state_dict[k] = v

    hf_model.load_state_dict(hf_state_dict, strict=False)
    hf_model.to(device)
    hf_model.eval()

    return hf_model

def print_result(name, results):
    """Print results immediately after evaluation."""
    print(f"\n{'='*70}")
    print(f"✓ {name} Results:")
    print(f"{'-'*70}")

    for task in tasks:
        if task in results:
            # Try different metric names
            score = None
            for metric in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc', 'mc1', 'mc2']:
                if metric in results[task]:
                    score = results[task][metric]
                    break

            if score is not None:
                print(f"  {task:<20} {score:.4f} ({score*100:.2f}%)")
            else:
                print(f"  {task:<20} Available: {list(results[task].keys())}")
        else:
            print(f"  {task:<20} Not found")

    print(f"{'='*70}\n")

# Store all results
all_results = {}

# --- 1. Evaluate Stock GPT-2 (The Control) ---
print("\n[1/3] Evaluating Stock GPT-2 (baseline reference)...")
print("-" * 70)
stock_model = AutoModelForCausalLM.from_pretrained('gpt2')
stock_model.to(device)
stock_lm = HFLM(pretrained=stock_model, tokenizer='gpt2', device=device, batch_size=4)

eval_out = simple_evaluate(
    model=stock_lm,
    tasks=tasks,
    limit=limit,
    random_seed=42, numpy_random_seed=42, torch_random_seed=42
)
all_results['Stock GPT-2'] = eval_out['results']
print_result('Stock GPT-2', eval_out['results'])
del stock_model

# --- 2. Evaluate Baseline ---
if os.path.exists('out/baseline/ckpt.pt'):
    print("\n[2/3] Evaluating Baseline (AdamW on noisy data)...")
    print("-" * 70)
    baseline_model = load_nanogpt_as_hf('out/baseline/ckpt.pt')
    baseline_lm = HFLM(pretrained=baseline_model, tokenizer='gpt2', device=device, batch_size=4)



    eval_out = simple_evaluate(
        model=baseline_lm,
        tasks=tasks,
        limit=limit,
        random_seed=42, numpy_random_seed=42, torch_random_seed=42
    )
    all_results['Baseline (AdamW)'] = eval_out['results']
    print_result('Baseline (AdamW)', eval_out['results'])
    del baseline_model
else:
    print("⚠️ Baseline checkpoint not found!")

# --- 3. Evaluate Sceptical ---
if os.path.exists('out/sceptical/ckpt.pt'):
    print("\n[3/3] Evaluating Sceptical (ScepticalAdam on noisy data)...")
    print("-" * 70)
    sceptical_model = load_nanogpt_as_hf('out/sceptical/ckpt.pt')
    sceptical_lm = HFLM(pretrained=sceptical_model, tokenizer='gpt2', device=device, batch_size=4)

    eval_out = simple_evaluate(
        model=sceptical_lm,
        tasks=tasks,
        limit=limit,
        random_seed=42, numpy_random_seed=42, torch_random_seed=42
    )
    all_results['Sceptical'] = eval_out['results']
    print_result('Sceptical', eval_out['results'])
    del sceptical_model
else:
    print("⚠️ Sceptical checkpoint not found!")

# --- 4. Final Comparison Table ---
print("\n" + "="*70)
print("FINAL COMPARISON - FACTUAL CORRECTNESS")
print("="*70)
print()

for task in tasks:
    print(f"\n{task.upper()}")
    print("-" * 70)
    print(f"{'Model':<25} {'Accuracy':<15}")
    print("-" * 70)

    for model_name, results in all_results.items():
        if task in results:
            score = None
            for metric in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc', 'mc1', 'mc2']:
                if metric in results[task]:
                    score = results[task][metric]
                    break

            if score is not None:
                print(f"{model_name:<25} {score:.4f} ({score*100:.2f}%)")
            else:
                print(f"{model_name:<25} N/A")

print("\n" + "="*70)
print("INTERPRETATION GUIDE")
print("="*70)
print("""
TruthfulQA MC1: Tests factual correctness (single correct answer)
  - Higher = Better at avoiding common misconceptions
  - Tests resistance to learning false information

TruthfulQA MC2: Multi-true variant (multiple correct answers possible)
  - Higher = Better at recognizing nuanced truths
  - More challenging than MC1

HellaSwag: Common sense reasoning
  - For comparison with previous results
  - Tests logical reasoning, not factual knowledge

EXPECTED RESULTS:
  If ScepticalAdam works as intended:
    - Sceptical should have HIGHER TruthfulQA scores (preserved facts)
    - Baseline may have LOWER TruthfulQA scores (learned noise)
    - HellaSwag may be similar (reasoning preserved in both)
""")
print("="*70)
