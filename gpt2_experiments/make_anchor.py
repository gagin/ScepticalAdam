"""
Generate Truth Vectors for ScepticalAdam optimizer.

This script:
1. Loads a pre-trained GPT-2 model
2. Loads a batch from the anchor dataset (high-quality educational content)
3. Performs a forward/backward pass to compute gradients
4. Normalizes the gradients to create unit vectors
5. Saves these as "truth_vectors.pt" for use by the ScepticalAdam optimizer
"""
import os
import torch
import numpy as np
from model import GPT

# Configuration
# Use MPS if available (Apple Silicon), otherwise CPU
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

batch_size = 4  # Reduced for 8GB RAM
block_size = 512  # Reduced from 1024 for memory efficiency
data_dir = 'data/anchor'
output_file = 'truth_vectors.pt'

print("="*60)
print("GENERATING TRUTH VECTORS")
print("="*60)
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"Block size: {block_size}")
print(f"Data directory: {data_dir}")
print()

# Load the pre-trained GPT-2 model
print("Loading GPT-2 (124M) model...")
model = GPT.from_pretrained('gpt2', override_args={'dropout': 0.0})

# Crop the model to the desired block size
if block_size < model.config.block_size:
    print(f"Cropping model block size from {model.config.block_size} to {block_size}...")
    model.crop_block_size(block_size)

model.to(device)
model.train()  # We need gradients
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Load anchor data
print("Loading anchor dataset...")
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
print(f"Anchor dataset size: {len(train_data):,} tokens")
print()

# Get a batch of data
def get_batch(split='train'):
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Get a batch from anchor data
print("Sampling batch from anchor data...")
X, Y = get_batch()
print(f"Batch shape: {X.shape}")
print()

# Forward pass
print("Performing forward pass...")
logits, loss = model(X, Y)
print(f"Loss: {loss.item():.4f}")
print()

# Backward pass to compute gradients
print("Performing backward pass...")
model.zero_grad()
loss.backward()
print("Gradients computed.")
print()

# Extract and normalize gradients
print("Extracting and normalizing gradients...")
truth_vectors = {}
total_params = 0
params_with_grad = 0

for name, param in model.named_parameters():
    total_params += 1
    if param.grad is not None:
        params_with_grad += 1
        # Clone the gradient and normalize it to a unit vector
        grad = param.grad.clone().detach()
        
        # Normalize: v_unit = v / ||v||
        grad_norm = torch.norm(grad)
        if grad_norm > 1e-8:
            grad_normalized = grad / grad_norm
        else:
            # If gradient is essentially zero, keep it as is
            grad_normalized = grad
        
        truth_vectors[name] = grad_normalized.cpu()
        
        if total_params <= 5:  # Show first few for debugging
            print(f"  {name}: shape={grad.shape}, norm={grad_norm:.6f}")

print()
print(f"Total parameters: {total_params}")
print(f"Parameters with gradients: {params_with_grad}")
print(f"Truth vectors created: {len(truth_vectors)}")
print()

# Save truth vectors
print(f"Saving truth vectors to {output_file}...")
torch.save(truth_vectors, output_file)
print("✓ Truth vectors saved successfully!")
print()

# Verify the saved file
print("Verifying saved file...")
loaded = torch.load(output_file)
print(f"  Loaded {len(loaded)} truth vectors")
print(f"  Sample keys: {list(loaded.keys())[:3]}")
print()

print("="*60)
print("TRUTH VECTOR GENERATION COMPLETE!")
print("="*60)
print(f"\nThe truth vectors have been saved to: {output_file}")
print("These will be used by ScepticalAdam to quarantine misaligned updates.")

