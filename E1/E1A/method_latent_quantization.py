# E1/E1A/method_latent_quantization.py
import argparse, json, torch, os
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from .bakeoff_utils import (
    get_dir_size,
    LSQDecoder,
    calculate_bpt,
    calculate_reconstruction_fidelity,
    align_tokens,
)


def train_decoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    encoder = AutoModelForMaskedLM.from_pretrained(args.model_path).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    decoder = LSQDecoder(encoder.config.hidden_size, encoder.config.vocab_size).to(
        device
    )
    optimizer = AdamW(decoder.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_dataset = dataset["train"].select(range(2000))
    val_dataset = dataset["validation"].select(range(500))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    for epoch in range(3):
        decoder.train()
        for batch in train_loader:
            texts = [t for t in batch["text"] if t]
            if not texts:
                continue

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                outputs = encoder.bert(
                    inputs.input_ids, attention_mask=inputs.attention_mask
                )
                latent_states = outputs[0]

            scale = (latent_states.max() - latent_states.min()) / 255
            quantized_states = torch.quantize_per_tensor(
                latent_states, scale, 0, torch.qint8
            )
            dequantized_states = quantized_states.dequantize()

            logits = decoder(dequantized_states)
            loss = loss_fn(
                logits.view(-1, encoder.config.vocab_size), inputs.input_ids.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                texts = [t for t in batch["text"] if t]
                if not texts:
                    continue
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
                outputs = encoder.bert(
                    inputs.input_ids, attention_mask=inputs.attention_mask
                )
                latent_states = outputs[0]
                scale = (latent_states.max() - latent_states.min()) / 255
                quantized_states = torch.quantize_per_tensor(
                    latent_states, scale, 0, torch.qint8
                )
                dequantized_states = quantized_states.dequantize()
                logits = decoder(dequantized_states)
                val_loss += loss_fn(
                    logits.view(-1, encoder.config.vocab_size),
                    inputs.input_ids.view(-1),
                ).item()

        print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader)}")

    os.makedirs(os.path.dirname(args.decoder_path), exist_ok=True)
    torch.save(decoder.state_dict(), args.decoder_path)
    torch.cuda.empty_cache()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    encoder = AutoModelForMaskedLM.from_pretrained(args.model_path).to(device)
    decoder = LSQDecoder(encoder.config.hidden_size, encoder.config.vocab_size).to(
        device
    )
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()

    dataset = load_dataset(args.dataset_name, args.dataset_config, split="test")
    sample_texts = [item["text"] for item in dataset.select(range(100)) if item["text"]]

    total_bits, total_tokens, total_fidelity = 0, 0, 0

    for text in sample_texts:
        original_tokens = tokenizer.tokenize(text)
        if not original_tokens:
            continue

        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = encoder.bert(inputs.input_ids)
            latent_states = outputs[0]

        scale = (latent_states.max() - latent_states.min()) / 255
        quantized_states = torch.quantize_per_tensor(
            latent_states, scale, 0, torch.qint8
        )
        compressed_bits = quantized_states.int_repr().numel() * 8

        dequantized_states = quantized_states.dequantize()
        with torch.no_grad():
            logits = decoder(dequantized_states)
            predicted_ids = logits.argmax(dim=-1)[0]

        decompressed_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
        decompressed_tokens = tokenizer.tokenize(decompressed_text)

        total_bits += compressed_bits
        total_tokens += len(original_tokens)
        total_fidelity += calculate_reconstruction_fidelity(
            original_tokens, decompressed_tokens
        )
        torch.cuda.empty_cache()

    avg_bpt = calculate_bpt(total_bits, total_tokens)
    avg_fidelity = total_fidelity / len(sample_texts) if sample_texts else 0
    model_size_mb = get_dir_size(args.model_path) + (
        os.path.getsize(args.decoder_path) / (1024 * 1024)
    )

    results = {
        "method": "Latent Space Quantization",
        "compression_efficiency_bpt": avg_bpt,
        "reconstruction_fidelity": avg_fidelity,
        "model_size_mb": model_size_mb,
    }
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--decoder-path", required=True)
    parser.add_argument("--train-decoder", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.train_decoder:
        train_decoder(args)
    elif args.evaluate:
        evaluate(args)


if __name__ == "__main__":
    main()
