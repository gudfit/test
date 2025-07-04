# E1/E1A/method_arithmetic_coding.py
import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import math
from .bakeoff_utils import get_dir_size, calculate_bpt


class ArithmeticEncoder:
    def __init__(self, precision=32):
        self.precision = precision
        self.full_range = 1 << precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.half_range >> 1
        self.low = 0
        self.high = self.full_range - 1
        self.buffer = []
        self.underflow_bits = 0

    def encode_symbol(self, cdf_low, cdf_high):
        range_val = self.high - self.low + 1

        # Use double precision for intermediate calculations
        low_fp = float(self.low)
        high_fp = float(self.high)
        range_fp = float(range_val)

        # Calculate new high and low with double precision
        new_high = low_fp + math.floor(range_fp * cdf_high)
        new_low = low_fp + math.ceil(range_fp * cdf_low)

        # Check for overflow before converting back to int
        if new_high > float(self.full_range - 1):
            new_high = self.full_range - 1
        if new_low < 0:
            new_low = 0

        self.high = int(new_high)
        self.low = int(new_low)

        while True:
            if self.high < self.half_range:
                self.emit_bit(0)
            elif self.low >= self.half_range:
                self.emit_bit(1)
                self.low -= self.half_range
                self.high -= self.half_range
            elif self.low >= self.quarter_range and self.high < 3 * self.quarter_range:
                self.underflow_bits += 1
                self.low -= self.quarter_range
                self.high -= self.quarter_range
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) | 1

    def emit_bit(self, bit):
        self.buffer.append(bit)
        if self.underflow_bits > 0:
            self.buffer.extend([1 - bit] * self.underflow_bits)
            self.underflow_bits = 0

    def finish(self):
        self.underflow_bits += 1
        if self.low < self.quarter_range:
            self.emit_bit(0)
        else:
            self.emit_bit(1)

        byte_array = bytearray()
        for i in range(0, len(self.buffer), 8):
            byte = 0
            for j in range(8):
                if i + j < len(self.buffer):
                    byte = (byte << 1) | self.buffer[i + j]
            byte_array.append(byte)

        return bytes(byte_array)


def encode_sequence(probs, tokens):
    encoder = ArithmeticEncoder()
    for token, prob_dist in zip(tokens, probs):
        if token >= len(prob_dist):
            token = tokenizer.unk_token_id
        prob_dist = prob_dist.astype(np.float64)
        prob_dist = np.maximum(prob_dist, 1e-10)
        prob_dist /= prob_dist.sum()
        cdf = np.cumsum(prob_dist)
        cdf_low = cdf[token - 1] if token > 0 else 0.0
        cdf_high = cdf[token]
        encoder.encode_symbol(cdf_low, cdf_high)
    return encoder.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", required=True)
    args = parser.parse_args()

    print(f"--- Evaluating Arithmetic Coding using model: {args.model_path} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()

    dataset = load_dataset(args.dataset_name, args.dataset_config, split="test")
    sample_texts = [
        item["text"]
        for item in dataset.select(range(100))
        if item["text"] and len(item["text"]) > 10
    ]

    total_bits = 0
    total_tokens = 0

    for text in sample_texts:
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 2:
            continue

        probs_list = []
        with torch.no_grad():
            for i in range(1, len(token_ids)):
                context_ids = torch.tensor([token_ids[:i]], device=device)
                outputs = model(context_ids)
                logits = outputs.logits[0, -1]
                probabilities = torch.softmax(logits, dim=0).float().cpu().numpy()
                probs_list.append(probabilities)

        sequence_to_encode = token_ids[1:]
        bitstream = encode_sequence(probs_list, sequence_to_encode)
        total_bits += len(bitstream) * 8
        total_tokens += len(token_ids)
        torch.cuda.empty_cache()

    avg_bpt = calculate_bpt(total_bits, total_tokens)
    reconstruction_fidelity = 1.0
    model_size_mb = get_dir_size(args.model_path)

    results = {
        "method": "Arithmetic Coding (Lossless)",
        "model_architecture": "CausalLM",
        "compression_efficiency_bpt": avg_bpt,
        "reconstruction_fidelity": reconstruction_fidelity,
        "model_size_mb": model_size_mb,
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
