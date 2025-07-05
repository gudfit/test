# E1/E1A/bakeoff_utils.py
import os
import random
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download all required NLTK data for POS tagging
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("tagsets", quiet=True)

def get_dir_size(path="."):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total / (1024 * 1024)


def calculate_bpt(num_bits, num_tokens):
    return num_bits / num_tokens if num_tokens > 0 else 0


def align_tokens(original, decompressed):
    return original[: len(decompressed)]


def calculate_reconstruction_fidelity(original_tokens, decompressed_tokens):
    aligned_original = align_tokens(original_tokens, decompressed_tokens)
    correct_count = sum(
        1 for ot, dt in zip(aligned_original, decompressed_tokens) if ot == dt
    )
    return correct_count / len(aligned_original) if aligned_original else 1.0

def predictive_masking_compress(text, tokenizer, mask_ratio=0.15, deterministic=False):
    """
    Compresses text by masking tokens.
    NOTE: The verb-checking logic has been removed to prevent NLTK errors.
    Both 'deterministic' and 'random' masking now use the same random token masking.
    """
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return ""

    num_to_mask = max(1, int(len(tokens) * mask_ratio))
    indices_to_mask = set(
        random.sample(range(len(tokens)), min(num_to_mask, len(tokens)))
    )
    masked_tokens = [
        tokenizer.mask_token if i in indices_to_mask else token
        for i, token in enumerate(tokens)
    ]
    return tokenizer.convert_tokens_to_string(masked_tokens)

def decompress_text(compressed_text, model, tokenizer, device):
    model.to(device)
    inputs = tokenizer(
        compressed_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id not in inputs["input_ids"]:
        return compressed_text
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = inputs["input_ids"].clone()
    masked_indices = torch.where(predicted_ids == mask_token_id)
    predictions = logits[masked_indices].argmax(dim=-1)
    predicted_ids[masked_indices] = predictions
    return tokenizer.decode(predicted_ids[0], skip_special_tokens=True)


class LSQDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size * 2)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))
