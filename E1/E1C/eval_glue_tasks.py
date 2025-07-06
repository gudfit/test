#!/usr/bin/env python3
# E1/E1C/eval_glue_tasks.py
import torch
import argparse
import json
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset

TASK_TO_KEYS = {
    "mrpc": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
    "rte": ("sentence1", "sentence2"),
}

def evaluate_model_on_glue(model_path, task_name, cache_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = load_dataset("glue", task_name, cache_dir=cache_dir)
    metric = evaluate.load("glue", task_name, cache_dir=cache_dir)
    num_labels = len(dataset["train"].features["label"].names)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        keys = TASK_TO_KEYS[task_name]
        if len(keys) == 2:
            return tokenizer(examples[keys[0]], examples[keys[1]], truncation=True, padding="max_length", max_length=512)
        else:
            return tokenizer(examples[keys[0]], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        cache_dir=cache_dir
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    model.to(device)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_datasets["validation"]

    training_args = TrainingArguments(
        output_dir=f"{cache_dir}/glue_training_{task_name}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"--- Fine-tuning and evaluating on {task_name.upper()} ---")
    trainer.train()
    eval_results = trainer.evaluate()
    
    key = "accuracy" if task_name != "mrpc" else "f1"
    return eval_results[f"eval_{key}"]

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a set of GLUE tasks.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    args = parser.parse_args()

    tasks = ["sst2", "mrpc", "rte"]
    final_results = {}

    for task in tasks:
        score = evaluate_model_on_glue(args.model_path, task, args.cache_dir)
        final_results[task] = score

    results_summary = {
        "model_path": args.model_path,
        "assessment": "GLUE Benchmark Performance",
        "details": final_results
    }
    print("\n--- GLUE Evaluation Summary ---")
    print(json.dumps(results_summary, indent=4))

if __name__ == "__main__":
    main()
