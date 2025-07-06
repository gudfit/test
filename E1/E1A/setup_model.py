#!/usr/bin/env python3
# E1/E1A/setup_model.py
import os
import argparse
import json
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--results-file", type=str, required=True)
    args = parser.parse_args()

    model_exists = False
    if os.path.isdir(args.output_path):
        has_config = os.path.exists(os.path.join(args.output_path, "config.json"))

        model_weight_files = ["pytorch_model.bin", "model.safetensors"]
        has_model_weights = any(os.path.exists(os.path.join(args.output_path, f)) for f in model_weight_files)

        if has_config and has_model_weights:
            model_exists = True

    if model_exists:
        print(f"Model already exists at {args.output_path}. Skipping training.")
        training_results = {
            "model": args.model_name,
            "training_completed": True,
            "status": "skipped (already exists)"
        }
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        with open(args.results_file, "a") as f:
            f.write(json.dumps(training_results) + "\n")
        return

    os.makedirs(args.output_path, exist_ok=True)

    training_results = {
        "model": args.model_name,
        "training_completed": True,
        "status": "training completed successfully"
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    is_mlm = "bert" in args.model_name.lower() or "roberta" in args.model_name.lower()

    if is_mlm:
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    if tokenizer.pad_token == '[PAD]':
        model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset(args.dataset_name, args.dataset_config)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    tokenized_datasets = tokenized_datasets.filter(
        lambda example: len(example["input_ids"]) == 512
    )

    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
        save_total_limit=1,
        save_steps=500,
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_path)

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "a") as f:
        f.write(json.dumps(training_results) + "\n")

    tokenizer.save_pretrained(args.output_path)
    print(f"Model and tokenizer saved to {args.output_path}")

if __name__ == "__main__":
    main()

