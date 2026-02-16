#!/usr/bin/env python3
"""
Qwen2 5M Parameter LLM - Training from Scratch
"""

import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    hidden_size: int = 256
    intermediate_size: int = 684
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        freqs = torch.outer(torch.arange(max_seq_len), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Qwen2Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
    
    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(value, q_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(attn, value).transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(out)


class Qwen2MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, x, attention_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        if attention_mask is None:
            seq_len = input_ids.size(1)
            attention_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)[None, None, :, :]
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.norm(x)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return logits, loss
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_p=0.9, eos_token_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumsum > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return input_ids


class TextDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dir = Path(data_dir)
        
        self.files = []
        logger.info(f"Scanning {self.data_dir.absolute()}...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir.absolute()}")
        
        all_paths = list(self.data_dir.rglob("*"))
        logger.info(f"Found {len(all_paths)} total paths")
        
        for path in all_paths:
            if path.is_file():
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if len(content.strip()) > 0:
                            self.files.append((path, content))
                except:
                    pass
        
        logger.info(f"Successfully read {len(self.files)} text files")
        
        self.samples = []
        for path, content in self.files:
            tokens = tokenizer.encode(content)
            if len(tokens) == 0:
                continue
            
            start = 0
            while start < len(tokens):
                end = min(start + max_length, len(tokens))
                chunk = tokens[start:end]
                if len(chunk) > 1:
                    self.samples.append(chunk)
                if end >= len(tokens):
                    break
                start += max_length // 2
        
        logger.info(f"Created {len(self.samples)} training samples")
        
        if len(self.samples) == 0:
            logger.warning("No data found! Creating dummy sample.")
            self.samples = [[tokenizer.eos_token_id] * 10]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        
        if len(tokens) >= self.max_length:
            input_ids = tokens[:self.max_length]
        else:
            input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {'input_ids': input_ids, 'labels': labels}


def train_tokenizer(data_dir: str, vocab_size: int = 8000, save_path: str = "./tokenizer"):
    logger.info("Training tokenizer...")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path.absolute()}")
    
    texts = []
    for path in data_path.rglob("*"):
        if path.is_file():
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if len(text.strip()) > 0:
                        texts.append(text)
            except:
                pass
    
    if len(texts) == 0:
        raise ValueError(f"No text files found in {data_dir}")
    
    logger.info(f"Training on {len(texts)} files")
    
    temp_file = "/tmp/train_text.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(texts))
    
    # Train tokenizer with proper post-processing
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<|endoftext|>", "<pad>"],
        show_progress=True
    )
    tokenizer.train([temp_file], trainer)
    
    # Set up proper decoder - use decoders.ByteLevel, not processors.ByteLevel
    tokenizer.decoder = decoders.ByteLevel()
    
    # Add post-processor to handle special tokens properly
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A",
        special_tokens=[
            ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
        ],
    )
    
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<pad>",
        unk_token="<|endoftext|>",
        clean_up_tokenization_spaces=True,
    )
    
    os.makedirs(save_path, exist_ok=True)
    wrapped.save_pretrained(save_path)
    return wrapped


def save_hf_format(model, tokenizer, output_dir: str):
    """Save model and tokenizer in HF-compatible format"""
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "max_position_embeddings": model.config.max_position_embeddings,
        "rope_theta": model.config.rope_theta,
        "rms_norm_eps": model.config.rms_norm_eps,
        "tie_word_embeddings": model.config.tie_word_embeddings,
        "torch_dtype": "float32",
        "transformers_version": "4.35.0",
        "use_cache": False,
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Saved to {output_dir}")


def train():
    config = ModelConfig()
    data_dir = "./data"
    output_dir = "./output"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    tokenizer_path = "./tokenizer"
    if os.path.exists(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = train_tokenizer(data_dir, vocab_size=config.vocab_size, save_path=tokenizer_path)
    
    config.vocab_size = len(tokenizer)
    logger.info(f"Vocab size: {config.vocab_size}")
    
    model = Qwen2ForCausalLM(config).to(device)
    logger.info(f"Parameters: {model.count_parameters():,} (~{model.count_parameters()/1e6:.1f}M)")
    
    dataset = TextDataset(data_dir, tokenizer, max_length=512)
    
    dataset_len = len(dataset)
    if dataset_len == 1:
        train_set = dataset
        val_set = dataset
        logger.info("Only 1 sample, using for both train and val")
    else:
        train_size = max(1, int(0.9 * dataset_len))
        val_size = dataset_len - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"Train: {len(train_set)}, Val: {len(val_set)}")
    
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    best_val = float('inf')
    num_epochs = 5  # Changed from 3 to 5
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            _, loss = model(input_ids, labels=labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {i}, Loss: {loss.item():.4f}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                _, loss = model(input_ids, labels=labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} complete. Val loss: {val_loss:.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            save_hf_format(model, tokenizer, os.path.join(output_dir, "best"))
    
    save_hf_format(model, tokenizer, output_dir)
    
    if os.path.exists("./tokenizer"):
        shutil.rmtree("./tokenizer")
        logger.info("Cleaned up temp tokenizer directory")


if __name__ == "__main__":
    train()
