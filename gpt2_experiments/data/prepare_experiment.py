"""
Data preparation script for the ScepticalAdam experiment.
Downloads and tokenizes two datasets:
1. Anchor (Signal): fineweb-edu - 10MB of high-quality educational content
2. Noise (Swamp): fineweb - 100MB of raw web data
"""
import os
import tiktoken
import numpy as np
from datasets import load_dataset

def prepare_dataset(dataset_name, split, max_bytes, output_dir, dataset_config=None):
    """
    Download, tokenize, and save a dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        max_bytes: Maximum bytes to process
        output_dir: Directory to save the tokenized data
        dataset_config: Optional dataset configuration
    """
    print(f"\n{'='*60}")
    print(f"Preparing {output_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Target size: {max_bytes / 1024 / 1024:.1f} MB")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset from HuggingFace...")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Collect text until we reach the target size
    collected_text = []
    current_bytes = 0
    
    print("Collecting and tokenizing text...")
    for i, example in enumerate(dataset):
        text = example['text']
        text_bytes = len(text.encode('utf-8'))
        
        if current_bytes + text_bytes > max_bytes:
            # Add partial text to reach exactly max_bytes
            remaining_bytes = max_bytes - current_bytes
            # Estimate characters needed (rough approximation)
            chars_needed = int(remaining_bytes * len(text) / text_bytes)
            collected_text.append(text[:chars_needed])
            current_bytes += remaining_bytes
            break
        
        collected_text.append(text)
        current_bytes += text_bytes
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1} examples, {current_bytes / 1024 / 1024:.2f} MB")
    
    print(f"\nCollected {current_bytes / 1024 / 1024:.2f} MB of text from {len(collected_text)} examples")
    
    # Join all text
    full_text = '\n\n'.join(collected_text)
    
    # Tokenize
    print("Tokenizing...")
    train_ids = enc.encode_ordinary(full_text)
    print(f"Total tokens: {len(train_ids):,}")
    
    # Create a small validation set (10% of data)
    n = len(train_ids)
    train_ids_final = train_ids[:int(n*0.9)]
    val_ids = train_ids[int(n*0.9):]
    
    print(f"Train tokens: {len(train_ids_final):,}")
    print(f"Val tokens: {len(val_ids):,}")
    
    # Save to binary files
    train_ids_array = np.array(train_ids_final, dtype=np.uint16)
    val_ids_array = np.array(val_ids, dtype=np.uint16)
    
    train_path = os.path.join(output_dir, 'train.bin')
    val_path = os.path.join(output_dir, 'val.bin')
    
    train_ids_array.tofile(train_path)
    val_ids_array.tofile(val_path)
    
    print(f"\nSaved to:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"✓ Complete!\n")

if __name__ == '__main__':
    # Get the data directory
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Prepare Anchor dataset (high-quality educational content)
    # 10MB of fineweb-edu
    anchor_dir = os.path.join(data_dir, 'anchor')
    prepare_dataset(
        dataset_name='HuggingFaceFW/fineweb-edu',
        split='train',
        max_bytes=10 * 1024 * 1024,  # 10 MB
        output_dir=anchor_dir,
        dataset_config='default'
    )
    
    # Prepare Noise dataset (raw web data)
    # 100MB of fineweb (without edu filter)
    noise_dir = os.path.join(data_dir, 'noise')
    prepare_dataset(
        dataset_name='HuggingFaceFW/fineweb',
        split='train',
        max_bytes=100 * 1024 * 1024,  # 100 MB
        output_dir=noise_dir,
        dataset_config='default'
    )
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nAnchor data: {anchor_dir}")
    print(f"Noise data: {noise_dir}")

