import torch
import argparse
import json
import time
from transformers import AutoTokenizer, AutoModel

def analyze_performance(model_path, cache_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory_footprint_mb = 0
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    model = AutoModel.from_pretrained(
        model_path,
        cache_dir=cache_dir
    ).to(device)
    
    if device == "cuda":
        memory_footprint_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    model.eval()
    
    sample_text = "This is a sample sentence to measure inference performance."
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

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
        "vram_footprint_mb": memory_footprint_mb,
        "avg_latency_ms": avg_latency_ms
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze model compute and memory performance.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    args = parser.parse_args()

    performance_results = analyze_performance(args.model_path, args.cache_dir)

    results = {
        "model_path": args.model_path,
        "assessment": "Compute and Memory Analysis",
        "details": performance_results
    }
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
