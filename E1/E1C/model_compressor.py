#!/usr/bin/env python3
# E1/E1C/model_compressor.py
import torch
import argparse
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.nn.utils import prune

def compress_model_with_magnitude_pruning(model_path, output_path, sparsity_ratio):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Applying {sparsity_ratio*100}% unstructured magnitude pruning...")
    
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))
    
    if not parameters_to_prune:
        raise ValueError("No linear layers found for pruning.")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity_ratio,
    )

    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    print(f"Saving pruned model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Apply magnitude pruning to a Hugging Face model.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--sparsity", type=float, required=True)
    args = parser.parse_args()

    compress_model_with_magnitude_pruning(args.model_path, args.output_path, args.sparsity)

if __name__ == "__main__":
    main()

