"""
fast_eval.py - Speed Run Evaluation (< 5 mins)
Usage: python fast_eval.py
"""
import torch
import os
import sys
from transformers import GPT2LMHeadModel, GPT2Config, AutoModelForCausalLM
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from model import GPT, GPTConfig # From nanoGPT

# 1. Configuration
device = 'cpu' # Keep on CPU for stability on M2
limit = 1000   # Quick preview (5 minutes)
tasks = ['hellaswag']

print(f"🚀 Starting Fast Evaluation (Limit: {limit} samples)")
print(f"Device: {device}")

def load_nanogpt_as_hf(ckpt_path):
    """Load nanoGPT checkpoint and convert to HuggingFace format."""
    print(f"Loading {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Fix compile prefix
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()

    # Convert to HuggingFace format
    print("Converting to HuggingFace format...")
    hf_config = GPT2Config(
        vocab_size=gptconf.vocab_size,
        n_positions=gptconf.block_size,
        n_embd=gptconf.n_embd,
        n_layer=gptconf.n_layer,
        n_head=gptconf.n_head,
        n_inner=4 * gptconf.n_embd,
        activation_function='gelu_new',
        resid_pdrop=gptconf.dropout,
        embd_pdrop=gptconf.dropout,
        attn_pdrop=gptconf.dropout,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    hf_model = GPT2LMHeadModel(hf_config)

    # Map weights
    hf_state_dict = {}
    for k, v in model.state_dict().items():
        if any(x in k for x in ['c_attn.weight', 'c_proj.weight', 'c_fc.weight']):
            hf_state_dict[k] = v.t()  # Transpose Conv1D to Linear
        elif k.startswith('transformer.') or k == 'lm_head.weight':
            hf_state_dict[k] = v

    hf_model.load_state_dict(hf_state_dict, strict=False)
    hf_model.to(device)
    return hf_model

results = {}

# --- 1. Evaluate Stock GPT-2 (The Control) ---
print("\nRunning Stock GPT-2 Eval...")
# We load manually to avoid lm_eval passing 'dtype' to the constructor
stock_model = AutoModelForCausalLM.from_pretrained('gpt2')
stock_model.to(device)
stock_lm = HFLM(pretrained=stock_model, tokenizer='gpt2', device=device, batch_size=4)

eval_out = simple_evaluate(
    model=stock_lm,
    tasks=tasks,
    limit=limit,
    random_seed=42, numpy_random_seed=42, torch_random_seed=42
)
results['stock'] = eval_out['results']

# Print result immediately
res = eval_out['results']
if 'hellaswag' in res:
    score = None
    for metric in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']:
        if metric in res['hellaswag']:
            score = res['hellaswag'][metric]
            break
    if score is not None:
        print(f"✓ Stock GPT-2 Result: {score:.4f} ({score*100:.2f}%)")

del stock_model

# --- 2. Evaluate Baseline ---
if os.path.exists('out/baseline/ckpt.pt'):
    baseline_model = load_nanogpt_as_hf('out/baseline/ckpt.pt')
    baseline_lm = HFLM(pretrained=baseline_model, tokenizer='gpt2', device=device, batch_size=4)

    print("\nRunning Baseline Eval...")
    eval_out = simple_evaluate(
        model=baseline_lm,
        tasks=tasks,
        limit=limit,
        random_seed=42, numpy_random_seed=42, torch_random_seed=42
    )
    results['baseline'] = eval_out['results']

    # Print result immediately
    res = eval_out['results']
    if 'hellaswag' in res:
        score = None
        for metric in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']:
            if metric in res['hellaswag']:
                score = res['hellaswag'][metric]
                break
        if score is not None:
            print(f"✓ Baseline Result: {score:.4f} ({score*100:.2f}%)")

    del baseline_model
else:
    print("⚠️ Baseline checkpoint not found!")

# --- 3. Evaluate Sceptical ---
if os.path.exists('out/sceptical/ckpt.pt'):
    sceptical_model = load_nanogpt_as_hf('out/sceptical/ckpt.pt')
    sceptical_lm = HFLM(pretrained=sceptical_model, tokenizer='gpt2', device=device, batch_size=4)

    print("\nRunning Sceptical Eval...")
    eval_out = simple_evaluate(
        model=sceptical_lm,
        tasks=tasks,
        limit=limit,
        random_seed=42, numpy_random_seed=42, torch_random_seed=42
    )
    results['sceptical'] = eval_out['results']

    # Print result immediately
    res = eval_out['results']
    if 'hellaswag' in res:
        score = None
        for metric in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']:
            if metric in res['hellaswag']:
                score = res['hellaswag'][metric]
                break
        if score is not None:
            print(f"✓ Sceptical Result: {score:.4f} ({score*100:.2f}%)")

    del sceptical_model
else:
    print("⚠️ Sceptical checkpoint not found!")

# --- 4. Print Comparison ---
print("\n" + "="*50)
print(f"{'Model':<20} {'HellaSwag (Acc)':<15}")
print("-" * 50)

for name, res in results.items():
    if 'hellaswag' in res:
        # Check for various metric names used by different lm_eval versions
        score = None
        for metric in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']:
            if metric in res['hellaswag']:
                score = res['hellaswag'][metric]
                break
        
        if score is not None:
            print(f"{name.capitalize():<20} {score:.4f}")
        else:
            print(f"{name.capitalize():<20} {list(res['hellaswag'].keys())}")

print("="*50)
