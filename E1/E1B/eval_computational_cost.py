#!/usr/bin/env python3
# E1/E1B/eval_computational_cost.py 
import torch
import argparse
import json
import time
from transformers import AutoModel, AutoTokenizer
from fvcore.nn import FlopCountAnalysis

def analyze_computational_cost(model_name, cache_dir):
    """
    Analyzes the computational cost of a model by measuring its FLOPs and latency.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    model.eval()

    sequence_length = 128
    sample_text = "This is a sample sentence to measure the computational cost."
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        max_length=sequence_length,
        padding="max_length",
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_tuple = (
        inputs["input_ids"],
        inputs.get("attention_mask"),
        inputs.get("token_type_ids"),
    )
    input_tuple = tuple(t for t in input_tuple if t is not None)
    flop_analyzer = FlopCountAnalysis(model, inputs=input_tuple)
    total_flops = flop_analyzer.total()
    gflops = total_flops / 1e9
    for _ in range(5):
        with torch.no_grad():
            _ = model(**inputs)

    num_runs = 50
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**inputs)
    end_time = time.time()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000

    return {
        "gflops_per_inference": gflops,
        "avg_latency_ms_per_inference": avg_latency_ms,
        "sequence_length": sequence_length
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze Model Computational Cost.")
    parser.add_argument("--model-name", type=str, required=True, help="Base model name.")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Directory for caching.")
    args = parser.parse_args()

    cost_results = analyze_computational_cost(args.model_name, args.cache_dir)

    results = {
        "model_name": args.model_name,
        "assessment": "Computational Cost Analysis",
        "details": cost_results
    }
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
