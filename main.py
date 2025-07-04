import os
import argparse
import random
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset as HFDataset

CONFIG = {
    "results_dir": "results",
    "output_model_dir": "models",
    "logs_dir": "logs",
    "train_epochs": 10,
    "train_batch_size": 8,
    "max_length": 512,
    "save_steps": [1, 2, 5, 10],
}


def prepare_dataset(args, tokenizer):
    """
    Loads and tokenizes data from either a local file or Hugging Face Hub.
    """

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=CONFIG["max_length"],
            padding="max_length",
        )

    if args.file_path:
        print(f"Loading data from local file: {args.file_path}")
        if not os.path.exists(args.file_path):
            raise FileNotFoundError(f"File not found: {args.file_path}")
        with open(args.file_path, "r", encoding="utf-8") as f:

            lines = [line.strip() for line in f if line.strip()]
            data_dict = {"text": lines}

        raw_dataset = HFDataset.from_dict(data_dict)

        split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
        tokenized_datasets = split_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        return tokenized_datasets, lines

    elif args.dataset_name:
        print(
            f"Loading dataset '{args.dataset_name}' with config '{args.dataset_config}'..."
        )
        full_dataset = load_dataset(args.dataset_name, args.dataset_config)
        tokenized_datasets = full_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        eval_split_name = "validation" if "validation" in tokenized_datasets else "test"
        raw_text_for_eval = [
            item["text"] for item in full_dataset[eval_split_name] if item["text"]
        ]
        return tokenized_datasets, raw_text_for_eval
    else:
        raise ValueError("You must provide either --file-path or --dataset-name.")


class SaveModelCallback(TrainerCallback):
    """A TrainerCallback to save the model at specified epochs."""

    def __init__(self, save_epochs, output_dir_prefix):
        self.save_epochs = set(save_epochs)
        self.output_dir_prefix = output_dir_prefix

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(round(state.epoch))
        if current_epoch in self.save_epochs:
            model_path = f"{self.output_dir_prefix}_epoch{current_epoch}"
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
                kwargs["model"].save_pretrained(model_path)
                kwargs["tokenizer"].save_pretrained(model_path)
                print(f"Model saved to {model_path} at epoch {current_epoch}")


def train_model(args):
    """Fine-tunes a model on the provided dataset."""
    print(f"--- Starting Training for {args.model_name} ---")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    tokenized_dataset, _ = prepare_dataset(args, tokenizer)

    if "train" not in tokenized_dataset:
        raise KeyError("Training requires a 'train' split in the dataset.")
    train_dataset = tokenized_dataset["train"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    safe_model_name = args.model_name.split("/")[-1]
    dataset_id = args.dataset_nickname if args.dataset_nickname else args.dataset_name
    output_dir_prefix = os.path.join(
        CONFIG["output_model_dir"], f"{safe_model_name}_{dataset_id}"
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(CONFIG["results_dir"], "training_output"),
        num_train_epochs=CONFIG["train_epochs"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        logging_dir=CONFIG["logs_dir"],
        logging_steps=100,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[SaveModelCallback(CONFIG["save_steps"], output_dir_prefix)],
        tokenizer=tokenizer,
    )

    trainer.train()
    print("--- Training Finished ---")


def get_dir_size(path="."):
    """Calculates the total size of a directory in megabytes (MB)."""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total / (1024 * 1024)


def calculate_bpt(original_text, compressed_text, tokenizer):
    """Calculates Bits Per Token (BPT)."""
    original_tokens = tokenizer.tokenize(original_text)
    compressed_size_bits = len(compressed_text.encode("utf-8")) * 8
    return compressed_size_bits / len(original_tokens) if original_tokens else 0.0


def calculate_reconstruction_fidelity(original_text, decompressed_text, tokenizer):
    """Calculates token-level accuracy."""
    original_tokens = tokenizer.tokenize(original_text)
    decompressed_tokens = tokenizer.tokenize(decompressed_text)
    correct_count = sum(
        1 for ot, dt in zip(original_tokens, decompressed_tokens) if ot == dt
    )
    return correct_count / len(original_tokens) if original_tokens else 1.0


def predictive_masking_compress(text, tokenizer, mask_ratio=0.15):
    """Compresses text by masking tokens."""
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return ""
    num_to_mask = max(1, int(len(tokens) * mask_ratio))
    indices_to_mask = set(
        random.sample(range(len(tokens)), min(num_to_mask, len(tokens)))
    )
    masked_tokens = [
        tokenizer.mask_token if i in indices_to_mask else token
        for i, token in enumerate(tokens)
    ]
    return tokenizer.convert_tokens_to_string(masked_tokens)


def decompress_text(compressed_text, model, tokenizer, device):
    """Decompresses text by predicting masked tokens."""
    model.to(device)
    inputs = tokenizer(
        compressed_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    mask_token_id = tokenizer.mask_token_id

    if mask_token_id not in inputs["input_ids"]:
        return compressed_text

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = inputs["input_ids"].clone()
    masked_indices = torch.where(predicted_ids == mask_token_id)
    predictions = logits[masked_indices].argmax(dim=-1)
    predicted_ids[masked_indices] = predictions

    return tokenizer.decode(predicted_ids[0], skip_special_tokens=True)


def run_full_analysis(args):
    """Runs the full evaluation suite on a given model."""
    print(f"\n--- Running Full Analysis ---")
    print(f"  Model Path: {args.model_path}")
    print(f"  Mask Ratio: {args.mask_ratio}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForMaskedLM.from_pretrained(args.model_path)
    except OSError:
        print(f"ERROR: Model not found at {args.model_path}.")
        return

    _, eval_texts = prepare_dataset(args, tokenizer)
    sample_texts = random.sample(eval_texts, min(100, len(eval_texts)))

    bpt_scores, fidelity_scores = [], []
    total_compressed_size_bytes = 0

    for text in sample_texts:
        if not text:
            continue
        compressed = predictive_masking_compress(text, tokenizer, args.mask_ratio)
        decompressed = decompress_text(compressed, model, tokenizer, device)

        bpt_scores.append(calculate_bpt(text, compressed, tokenizer))
        fidelity_scores.append(
            calculate_reconstruction_fidelity(text, decompressed, tokenizer)
        )
        total_compressed_size_bytes += len(compressed.encode("utf-8"))

    avg_bpt = sum(bpt_scores) / len(bpt_scores) if bpt_scores else 0

    avg_fidelity = sum(fidelity_scores) / len(fidelity_scores) if fidelity_scores else 0

    model_size_mb = get_dir_size(args.model_path)
    avg_compressed_data_mb = (
        (total_compressed_size_bytes / len(sample_texts)) / (1024 * 1024)
        if sample_texts
        else 0
    )
    total_size_mb = model_size_mb + avg_compressed_data_mb

    results = {
        "model_path": args.model_path,
        "mask_ratio": args.mask_ratio,
        "compression_efficiency_bpt": avg_bpt,
        "reconstruction_fidelity": avg_fidelity,
        "model_size_mb": model_size_mb,
        "total_system_size_mb": total_size_mb,
    }

    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Compression Research Engine.")
    parser.add_argument(
        "--train", action="store_true", help="Run the training pipeline."
    )
    parser.add_argument(
        "--run-analysis", action="store_true", help="Run the full evaluation suite."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Base model from Hugging Face Hub.",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        help="Path to a local .txt file for training/evaluation.",
    )
    parser.add_argument(
        "--dataset-name", type=str, help="Dataset name from Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset-config", type=str, help="Config for the Hugging Face dataset."
    )
    parser.add_argument(
        "--dataset-nickname",
        type=str,
        help="A short name for the dataset (e.g., Othello) used in model save paths.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to a fine-tuned model directory for analysis.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for compression.",
    )

    args = parser.parse_args()

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    os.makedirs(CONFIG["output_model_dir"], exist_ok=True)

    if args.train:
        train_model(args)

    if args.run_analysis:
        if not args.model_path:
            parser.error("--run-analysis requires --model-path.")
        run_full_analysis(args)

    if not args.train and not args.run_analysis:
        parser.print_help()


if __name__ == "__main__":
    main()
