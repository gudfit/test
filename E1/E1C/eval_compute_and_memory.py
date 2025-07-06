#!/usr/bin/env python3
# E1/E1C/eval_compute_and_memory.py
import torch
import argparse
import json
import time
import os
from transformers import AutoTokenizer, AutoModel

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

def analyze_performance(model_path, cache_dir, sample_file, sample_length):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    sample_text = load_sample_text(sample_file, sample_length)
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=False).to(device)
    model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir).to(device)
    model.eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        with torch.no_grad():
            _ = model(**inputs)
    num_runs = 50
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**inputs)
    avg_latency_ms = ((time.time() - start) / num_runs) * 1000
    if device == "cuda":
        memory_footprint_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        memory_footprint_mb = 0
    return {"vram_footprint_mb": memory_footprint_mb, "avg_latency_ms": avg_latency_ms}

def default_sample_path():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "data", "wikipedia.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--sample-file", type=str, default=default_sample_path())
    parser.add_argument("--sample-length", type=int, default=1024)
    args = parser.parse_args()
    details = analyze_performance(args.model_path, args.cache_dir, args.sample_file, args.sample_length)
    print(json.dumps({"model_path": args.model_path, "assessment": "Compute and Memory Analysis", "details": details}, indent=4))

if __name__ == "__main__":
    main()

