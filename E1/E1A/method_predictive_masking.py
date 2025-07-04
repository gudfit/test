# E1/E1A/method_predictive_masking.py
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from .bakeoff_utils import (
    get_dir_size,
    predictive_masking_compress,
    decompress_text,
    calculate_bpt,
    calculate_reconstruction_fidelity,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--mask-ratio", type=float, required=True)
    parser.add_argument(
        "--masking-type",
        type=str,
        default="random",
        choices=["random", "deterministic"],
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_path).to(device)

    dataset = load_dataset(args.dataset_name, args.dataset_config, split="test")
    sample_texts = [item["text"] for item in dataset.select(range(100)) if item["text"]]

    total_bits, total_tokens, total_fidelity = 0, 0, 0

    for text in sample_texts:
        original_tokens = tokenizer.tokenize(text)
        if not original_tokens:
            continue

        compressed_text = predictive_masking_compress(
            text,
            tokenizer,
            args.mask_ratio,
            deterministic=(args.masking_type == "deterministic"),
        )

        decompressed_text = decompress_text(compressed_text, model, tokenizer, device)
        decompressed_tokens = tokenizer.tokenize(decompressed_text)

        total_bits += len(compressed_text.encode("utf-8")) * 8
        total_tokens += len(original_tokens)
        total_fidelity += calculate_reconstruction_fidelity(
            original_tokens, decompressed_tokens
        )

        # Clear GPU memory
        model.to("cpu")
        torch.cuda.empty_cache()
        model.to(device)

    avg_bpt = calculate_bpt(total_bits, total_tokens)
    avg_fidelity = total_fidelity / len(sample_texts) if sample_texts else 0
    model_size_mb = get_dir_size(args.model_path)

    results = {
        "method": "Predictive Masking",
        "model": args.model_path.split("/")[-1],
        "masking_type": args.masking_type,
        "mask_ratio": args.mask_ratio,
        "compression_efficiency_bpt": avg_bpt,
        "reconstruction_fidelity": avg_fidelity,
        "model_size_mb": model_size_mb,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
