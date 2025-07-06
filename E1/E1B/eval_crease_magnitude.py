# E1/E1B/eval_crease_magnitude.py
import os
import torch
import argparse
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('.')
from E1.E1A.bakeoff_utils import predictive_masking_compress, decompress_text, AutoModelForMaskedLM

def get_surprisal_vector(text, model, tokenizer, device):
    """Calculates the surprisal for each token in a text."""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(inputs.input_ids, labels=inputs.input_ids)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        surprisal_per_token = loss.view(shift_labels.size())
        return surprisal_per_token.cpu().numpy().flatten()


def calculate_crease_magnitude(original_text, oracle_model_name, compression_model_name, mask_ratio, cache_dir):
    """
    Calculates the 'Crease Magnitude' between an original text and its
    lossy reconstruction, based on the 'Crumpled Paper' analogy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name, cache_dir=os.path.join(cache_dir, "models"))
    oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name, cache_dir=os.path.join(cache_dir, "models")).to(device)
    oracle_model.eval()
    comp_tokenizer = AutoTokenizer.from_pretrained(compression_model_name, cache_dir=os.path.join(cache_dir, "models"))
    comp_model = AutoModelForMaskedLM.from_pretrained(compression_model_name, cache_dir=os.path.join(cache_dir, "models"))
    compressed_text = predictive_masking_compress(original_text, comp_tokenizer, mask_ratio=mask_ratio)
    reconstructed_text = decompress_text(compressed_text, comp_model, comp_tokenizer, device="cpu") 
    s_original = get_surprisal_vector(original_text, oracle_model, oracle_tokenizer, device)
    s_reconstructed = get_surprisal_vector(reconstructed_text, oracle_model, oracle_tokenizer, device)
    min_len = min(len(s_original), len(s_reconstructed))
    s_original_aligned = s_original[:min_len]
    s_reconstructed_aligned = s_reconstructed[:min_len]
    
    crease_magnitude = np.linalg.norm(s_reconstructed_aligned - s_original_aligned)
    
    return crease_magnitude, reconstructed_text

def main():
    parser = argparse.ArgumentParser(description="Calculate Crease Magnitude.")
    parser.add_argument("--compression-model", type=str, default="bert-base-cased", help="Masked LM for compression.")
    parser.add_argument("--oracle-model", type=str, default="gpt2", help="Causal LM to act as the oracle.")
    parser.add_argument("--mask-ratio", type=float, default=0.5, help="Mask ratio for lossy compression.")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Directory for caching.")
    args = parser.parse_args()

    original_text = (
        "In theoretical physics, quantum field theory is a theoretical framework that combines classical field theory, "
        "special relativity, and quantum mechanics. QFT is used in particle physics to construct physical models of "
        "subatomic particles and in condensed matter physics to construct models of quasiparticles."
    )

    cm_score, reconstructed = calculate_crease_magnitude(
        original_text, args.oracle_model, args.compression_model, args.mask_ratio, args.cache_dir
    )

    results = {
        "assessment": "Fidelity Degradation Profile (Crease Magnitude)",
        "compression_model": args.compression_model,
        "oracle_model": args.oracle_model,
        "mask_ratio": args.mask_ratio,
        "crease_magnitude": cm_score,
        "original_text": original_text,
        "reconstructed_text": reconstructed,
    }
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
