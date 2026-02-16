#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import math
from pathlib import Path
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

def train_tokenizer(text_files, vocab_size: int, out_dir: str):
    print(f"Training tokenizer on {text_files}...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=text_files,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_model(out_dir)

    tokenizer_hf = GPT2TokenizerFast.from_pretrained(out_dir)
    tokenizer_hf.pad_token = "<pad>"
    tokenizer_hf.bos_token = "<s>"
    tokenizer_hf.eos_token = "</s>"
    tokenizer_hf.unk_token = "<unk>"
    tokenizer_hf.mask_token = "<mask>"
    
    tokenizer_hf.save_pretrained(out_dir)
    return tokenizer_hf

def calculate_auto_epochs(num_samples, batch_size):
    """
    Logic to decide epochs:
    - We want a minimum of ~2,000 total steps for the model to learn anything.
    - We want a maximum of ~50,000 steps to avoid over-training for this script.
    - We cap epochs between 1 and 100.
    """
    steps_per_epoch = max(1, num_samples // batch_size)
    
    # Target approximately 3,000 total optimization steps
    target_steps = 3000
    suggested_epochs = math.ceil(target_steps / steps_per_epoch)
    
    # Apply constraints
    if num_samples < 100:
        epochs = 100 # Tiny dataset, needs many passes
    elif num_samples > 500000:
        epochs = 1   # Huge dataset, one pass is plenty
    else:
        epochs = max(3, min(suggested_epochs, 50)) # Between 3 and 50
        
    return epochs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.txt", help="Path to your training text file")
    parser.add_argument("--out_dir", type=str, default="my_model_hf")
    parser.add_argument("--epochs", type=int, default=None, help="Force specific epochs (overrides auto)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    # === 1. Train Tokenizer ===
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found.")
        return

    tokenizer = train_tokenizer([args.data], args.vocab_size, args.out_dir)

    # === 2. Load Data ===
    print(f"Loading text from {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    num_lines = len(lines)
    print(f"Found {num_lines} lines/chunks of text.")

    # === 3. Model Configuration ===
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)

    # === 4. Tokenization ===
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.seq_len)

    ds = Dataset.from_dict({"text": lines})
    tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === 5. Dynamic Epoch Calculation ===
    if args.epochs is not None:
        final_epochs = args.epochs
        print(f"Using user-defined epochs: {final_epochs}")
    else:
        final_epochs = calculate_auto_epochs(len(tokenized_ds), args.batch_size)
        print(f"Auto-calculated epochs based on dataset size: {final_epochs}")

    # === 6. Training ===
    print(f"Starting training on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")
    
    training_args = TrainingArguments(
        output_dir=f"{args.out_dir}/checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=final_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        eval_strategy="no",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_ds,
    )

    trainer.train()

    # === 7. Final Save ===
    print(f"\nSaving final model + config to {args.out_dir}...")
    model.config.model_type = "gpt2"
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    
    print("DONE!")

if __name__ == "__main__":
    main()
