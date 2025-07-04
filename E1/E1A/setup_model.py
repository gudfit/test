# E1/E1A/setup_model.py
import os
import argparse
import json
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import evaluate
import torch


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    if len(predictions) > 0:
        return accuracy_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--results-file", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    training_results = {
        "model": args.model_name,
        "epoch_accuracies": [],
        "final_accuracy": 0.0,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    is_mlm = "gpt" not in args.model_name.lower()

    if is_mlm:
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    dataset = load_dataset(args.dataset_name, args.dataset_config)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512, padding="max_length"
        )

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    tokenized_datasets = tokenized_datasets.filter(
        lambda example: len(example["input_ids"]) == 512
    )

    if "validation" not in tokenized_datasets:
        train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.1)
        tokenized_datasets["train"] = train_test_split["train"]
        tokenized_datasets["validation"] = train_test_split["test"]

    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
        save_total_limit=1,
        save_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics if is_mlm else None,
    )

    train_result = trainer.train()
    metrics = train_result.metrics

    epoch_accuracies = []
    for epoch in range(3):
        print(f"\nEvaluating after epoch {epoch+1}")
        eval_metrics = trainer.evaluate()
        accuracy = eval_metrics.get("eval_accuracy", 0.0)
        epoch_accuracies.append({"epoch": epoch + 1, "accuracy": accuracy})
        print(f"Epoch {epoch+1} Accuracy: {accuracy:.4f}")

        checkpoint_dir = f"./training_output/checkpoint-{epoch+1}"
        trainer.save_model(checkpoint_dir)

    trainer.save_model(args.output_path)

    training_results["epoch_accuracies"] = epoch_accuracies
    training_results["final_accuracy"] = epoch_accuracies[-1]["accuracy"]

    with open(args.results_file, "a") as f:
        f.write(json.dumps(training_results) + "\n")

    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
