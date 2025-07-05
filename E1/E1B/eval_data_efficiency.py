# E1/E1B/eval_data_efficiency.py
import torch
import argparse
import json
import time
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score

def evaluate_data_efficiency(model_name, task_name, cache_dir):
    """
    Measures data efficiency by fine-tuning on subsets of a downstream task.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(task_name, cache_dir=cache_dir)
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding="max_length", truncation=True)

    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # Updated to the more granular list of sample sizes
    data_subsets = [100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000]
    
    target_accuracy = 0.85
    results = {}

    for num_samples in data_subsets:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, cache_dir=cache_dir).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id

        # Ensure we don't request more samples than available in the dataset
        if num_samples > len(train_dataset):
            print(f"Warning: Requested {num_samples} samples, but dataset only has {len(train_dataset)}. Skipping.")
            continue

        subset_train_dataset = train_dataset.select(range(num_samples)).map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=f"{cache_dir}/training_output_{num_samples}",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_steps=50,
            report_to="none",
            save_total_limit=1,
            do_train=True,
            do_eval=True,
        )
        
        def compute_metrics(p):
            return {"accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=subset_train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        eval_results = trainer.evaluate()
        accuracy = eval_results['eval_accuracy']

        results[num_samples] = {
            "accuracy": accuracy,
            "training_time_seconds": training_time
        }
        
        if 'examples_to_reach_target' not in results and accuracy >= target_accuracy:
            results['examples_to_reach_target'] = num_samples
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Data Efficiency for Fine-Tuning.")
    parser.add_argument("--model-name", type=str, required=True, help="Base model name.")
    parser.add_argument("--task-name", type=str, default="sst2", help="Downstream task name.")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Directory for caching.")
    args = parser.parse_args()

    efficiency_results = evaluate_data_efficiency(args.model_name, args.task_name, args.cache_dir)
    
    final_results = {
        "model_name": args.model_name,
        "assessment": "Data Efficiency",
        "task": args.task_name,
        "details": efficiency_results
    }
    print(json.dumps(final_results, indent=4))

if __name__ == "__main__":
    main()
