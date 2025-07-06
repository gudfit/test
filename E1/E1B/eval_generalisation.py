#!/usr/bin/env python3
# E1/E1B/eval_generalisation.py
import os
import torch
import argparse
import json
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def calculate_perplexity(model_name, dataset_name, dataset_config, cache_dir):
    """
    Calculates the perplexity of a model on a given dataset.
    Note: This uses a CausalLM for straightforward PPL calculation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=os.path.join(cache_dir, "models")).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.join(cache_dir, "models"))
    model.eval()

    dataset = load_dataset(dataset_name, dataset_config, split='test', cache_dir=os.path.join(cache_dir, "datasets"))
    
    text = "\n\n".join(dataset.select(range(50))['text'])
    encodings = tokenizer(text, return_tensors='pt')

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    with torch.no_grad():
        for i in range(0, seq_len, stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, seq_len)
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model Generalisation via Perplexity.")
    
    parser.add_argument("--model-name", type=str, default="gpt2", help="Hugging Face model name for PPL calculation.")
    parser.add_argument("--dataset-name", type=str, default="wikitext", help="OOD dataset name.")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="OOD dataset configuration.")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Directory for caching.")
    args = parser.parse_args()

    perplexity = calculate_perplexity(args.model_name, args.dataset_name, args.dataset_config, args.cache_dir)
    
    results = {
        "model_name": args.model_name,
        "assessment": "Generalisation to Unseen Domain",
        "perplexity": perplexity
    }
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
