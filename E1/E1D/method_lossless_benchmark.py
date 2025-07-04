# E1/E1D/method_lossless_benchmark.py
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch_ac
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a CausalLM for lossless compression."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Hugging Face model ID (e.g., 'mistralai/Mistral-7B-v0.1').",
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", required=True)
    args = parser.parse_args()

    print(f"--- Benchmarking {args.model_name} on {args.dataset_name} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    dataset = load_dataset(args.dataset_name, args.dataset_config, split="test")

    total_bits = 0
    total_chars = 0

    print("Encoding dataset with model probabilities...")
    for text in tqdm(dataset["text"], desc="Compressing"):
        if not text:
            continue
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 2:
            continue

        total_chars += len(text.encode("utf-8"))

        cdf_list = []
        with torch.no_grad():
            for i in range(1, len(token_ids)):
                context_ids = torch.tensor([token_ids[:i]], device=device)
                logits = model(context_ids).logits[0, -1]
                probs = torch.softmax(logits, dim=0)
                cdf = torch.cumsum(probs, dim=0)
                cdf = torch.cat([torch.tensor([0.0], device=device), cdf], dim=0)
                cdf_list.append(cdf.cpu())

        sequence_to_encode = torch.tensor(token_ids[1:])
        bitstream = torch_ac.encode_float_cdf(
            cdf_list, sequence_to_encode, needs_normalization=True
        )
        total_bits += len(bitstream) * 8

    bpc = total_bits / total_chars if total_chars > 0 else 0

    results = {
        "method": f"LLM + Arithmetic Coding ({args.model_name})",
        "bits_per_character": bpc,
    }

    print(f"RESULT: {results['method']}: {results['bits_per_character']:.4f} BPC")


if __name__ == "__main__":
    main()
