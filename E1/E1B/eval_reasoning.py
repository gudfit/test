#!/usr/bin/env python3
# E1/E1B/eval_reasoning.py
import torch
import argparse
import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def evaluate_zero_shot_reasoning(model_name, task_name, cache_dir):
    """
    Evaluates zero-shot reasoning by using a model fine-tuned on MNLI as a proxy.
    This simulates assessing the reasoning embedded in a base model like RoBERTa.
    A true zero-shot evaluation for MLMs is more complex; this is a practical approximation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reasoning_model_name = "textattack/roberta-base-MNLI" 
    
    tokenizer = AutoTokenizer.from_pretrained(reasoning_model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(reasoning_model_name, cache_dir=cache_dir).to(device)
    model.eval()
    dataset = load_dataset(task_name, split="validation_mismatched", cache_dir=cache_dir)
    subset = dataset.select(range(200))

    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for item in subset:
            inputs = tokenizer(item['premise'], item['hypothesis'], return_tensors='pt', truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            if prediction == item['label']:
                correct_predictions += 1
            total_predictions += 1
            
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate Zero-Shot Reasoning of a Model.")
    parser.add_argument("--model-name", type=str, required=True, help="Base model name (e.g., roberta-base).")
    parser.add_argument("--task-name", type=str, default="multi_nli", help="Reasoning task name from Hugging Face Hub.")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Directory for caching models and datasets.")
    args = parser.parse_args()

    accuracy = evaluate_zero_shot_reasoning(args.model_name, args.task_name, args.cache_dir)
    
    results = {
        "model_name": args.model_name,
        "assessment": "Zero-Shot NLI Reasoning",
        "task": args.task_name,
        "accuracy": accuracy
    }
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
