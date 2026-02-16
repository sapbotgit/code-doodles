#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
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
    print(f"Training tokenizer on {len(text_files)} files...")
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

    # Custom chat template for llama.cpp
    tokenizer_hf.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "# QUESTION\n{{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "# ANSWER\n{{ message['content'] }}</s>"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "# ANSWER\n"
        "{% endif %}"
    )
    tokenizer_hf.save_pretrained(out_dir)
    return tokenizer_hf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled", type=str, default="unlabeled.txt")
    parser.add_argument("--labeled", type=str, default="labeled.txt")
    parser.add_argument("--out_dir", type=str, default="my_model_hf")
    parser.add_argument("--epochs_pre", type=int, default=3)
    parser.add_argument("--epochs_ft", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    # === 1. Train tokenizer ===
    files_for_tokenizer = []
    for f in [args.unlabeled, args.labeled]:
        if os.path.exists(f):
            files_for_tokenizer.append(f)
    if not files_for_tokenizer:
        print("No data files!")
        return

    tokenizer = train_tokenizer(files_for_tokenizer, args.vocab_size, args.out_dir)

    # === 2. Load data ===
    def load_texts(path):
        if not os.path.exists(path):
            return []
        text = Path(path).read_text(encoding="utf-8")
        chunks = [c.strip() + "</s>" for c in text.split("</s>") if c.strip()]
        print(f"   → {len(chunks)} chunks from {path}")
        return chunks

    def load_qa(path):
        if not os.path.exists(path):
            return []
        blocks = Path(path).read_text(encoding="utf-8").split("# QUESTION")
        entries = []
        for b in blocks:
            if b.strip() and "# ANSWER" in b:
                entry = "# QUESTION" + b.strip()
                if not entry.endswith("</s>"):
                    entry += "</s>"
                entries.append(entry)
        print(f"   → {len(entries)} Q/A pairs from {path}")
        return entries

    unlabeled = load_texts(args.unlabeled)
    labeled = load_qa(args.labeled)

    # === 3. Model & Config ===
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

    # === 4. Tokenize function (NO padding here!) ===
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.seq_len)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === 5. Phase 1: Unlabeled pretraining ===
    if unlabeled:
        print(f"\nPhase 1: Pretraining on {len(unlabeled)} unlabeled chunks...")
        ds = Dataset.from_dict({"text": unlabeled}).map(tokenize_fn, batched=True, remove_columns=["text"])
        
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"{args.out_dir}/pretrain",
                per_device_train_batch_size=args.batch_size,
                num_train_epochs=args.epochs_pre,
                learning_rate=args.lr,
                logging_steps=10,
                save_steps=1000,
                save_total_limit=2,
                fp16=torch.cuda.is_available(),
                warmup_steps=50,
                weight_decay=0.01,
                eval_strategy="no",           # ← FIXED: was evaluation_strategy
                disable_tqdm=False,
            ),
            data_collator=data_collator,
            train_dataset=ds,
        )
        trainer.train()
        model = trainer.model

    # === 6. Phase 2: Fine-tuning on Q/A ===
    if labeled:
        print(f"\nPhase 2: Fine-tuning on {len(labeled)} Q/A pairs...")
        ds = Dataset.from_dict({"text": labeled}).map(tokenize_fn, batched=True, remove_columns=["text"])
        
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"{args.out_dir}/finetune",
                per_device_train_batch_size=args.batch_size // 2,
                gradient_accumulation_steps=4,
                num_train_epochs=args.epochs_ft,
                learning_rate=3e-4,
                logging_steps=5,
                save_steps=500,
                fp16=torch.cuda.is_available(),
                warmup_steps=50,
                eval_strategy="no",           # ← FIXED
            ),
            data_collator=data_collator,
            train_dataset=ds,
        )
        trainer.train()

    # === 7. FINAL SAVE (THIS FIXES config.json!) ===
    print(f"\nSaving final model + config to {args.out_dir}...")
    model.config.model_type = "gpt2"           # ← THIS LINE IS CRITICAL!
    model.config.save_pretrained(args.out_dir) # ← Save config properly
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print("DONE! Now convert with:")
    print(f"python llama.cpp/convert_hf_to_gguf.py {args.out_dir} --outfile romgpt.gguf --outtype q8_0")

if __name__ == "__main__":
    main()
