#!/usr/bin/env python3
# E1/E1B/generate_probes.py
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model on a local text file.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True, help="Path to the local text file for training.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the fine-tuned model.")
    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading model {args.model_name}...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    print(f"Creating dataset from {args.train_file}...")
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.train_file,
        block_size=128,  # Adjust block size based on your GPU memory
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"./training_output_{args.model_name.replace('/', '_')}",
        overwrite_output_dir=True,
        num_train_epochs=1, # Keep it to 1 epoch for faster fine-tuning
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        report_to="none", # Suppress wandb logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Fine-tuning complete. Saving model to {args.output_path}")
    trainer.save_model(args.output_path)
    tokenizer.save_pretrained(args.output_path)

if __name__ == "__main__":
    main()
