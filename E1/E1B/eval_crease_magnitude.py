#!/usr/bin/env python3
# E1/E1B/eval_crease_magnitude.py
import torch
import argparse
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
import sys

sys.path.append('.')
from E1.E1A.bakeoff_utils import predictive_masking_compress, decompress_text

def load_sample_text(path, length):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                words.extend(line.split())
            if len(words) >= length:
                break
    return " ".join(words[:length])

def get_surprisal_vector(text, model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        if inputs.input_ids.shape[1] == 0:
            return np.array([])
        outputs = model(inputs.input_ids, labels=inputs.input_ids)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss.view(shift_labels.size()).cpu().numpy().flatten()

def calculate_crease_magnitude(original_text, oracle_model_name, compression_model_name, mask_ratio, cache_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name, cache_dir=cache_dir)
    oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name, cache_dir=cache_dir).to(device)
    oracle_model.eval()
    comp_tokenizer = AutoTokenizer.from_pretrained(compression_model_name, cache_dir=cache_dir)
    comp_model = AutoModelForMaskedLM.from_pretrained(compression_model_name, cache_dir=cache_dir)
    compressed_text = predictive_masking_compress(original_text, comp_tokenizer, mask_ratio=mask_ratio)
    reconstructed_text = decompress_text(compressed_text, comp_model, comp_tokenizer, "cpu")
    s_original = get_surprisal_vector(original_text, oracle_model, oracle_tokenizer, device)
    s_reconstructed = get_surprisal_vector(reconstructed_text, oracle_model, oracle_tokenizer, device)
    min_len = min(len(s_original), len(s_reconstructed))
    if min_len == 0:
        return 0.0
    return float(np.linalg.norm(s_reconstructed[:min_len] - s_original[:min_len]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression-model", type=str, required=True)
    parser.add_argument("--oracle-model", type=str, default="gpt2")
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--original-file", type=str, default="../../data/wikipedia.txt")
    parser.add_argument("--sample-length", type=int, default=2048)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    args = parser.parse_args()
    original_text = load_sample_text(args.original_file, args.sample_length)
    cm_score = calculate_crease_magnitude(original_text, args.oracle_model, args.compression_model, args.mask_ratio, args.cache_dir)
    results = {
        "assessment": "Fidelity Degradation Profile (Crease Magnitude)",
        "compression_model": args.compression_model,
        "oracle_model": args.oracle_model,
        "mask_ratio": args.mask_ratio,
        "crease_magnitude": cm_score
    }
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()

