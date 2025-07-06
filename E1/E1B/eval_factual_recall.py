# E1/E1B/eval_factual_recall.py
import torch
import argparse
import json
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer
from knowledge_quant_utils import get_dir_size

def evaluate_factual_recall(model_path, probe_file):
    """
    Evaluates factual recall using a fine-tuned model and a local probe file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
    model.eval()

    model_size_mb = get_dir_size(model_path)

    with open(probe_file, 'r') as f:
        factual_probes = json.load(f)

    correct_predictions = 0
    with torch.no_grad():
        for probe in factual_probes:
            
            template = probe["template"].replace("[MASK]", tokenizer.mask_token)
            
            inputs = tokenizer(template, return_tensors="pt").to(device)
            mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
            
            logits = model(**inputs).logits
            predicted_token_id = logits[0, mask_token_index, :].argmax(dim=-1)
            predicted_token = tokenizer.decode(predicted_token_id).strip()

            if predicted_token.lower() == probe["answer"].lower():
                correct_predictions += 1

    recall = correct_predictions / len(factual_probes) if factual_probes else 0
    return recall, model_size_mb

def main():
    parser = argparse.ArgumentParser(description="Evaluate Factual Recall from a local model and probe file.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--probe-file", type=str, required=True, help="Path to the JSON file with probes.")
    args = parser.parse_args()

    recall_score, model_size = evaluate_factual_recall(args.model_path, args.probe_file)
    
    results = {
        "model_path": args.model_path,
        "assessment": "Factual Recall from Local Data",
        "model_size_mb": model_size,
        "factual_recall_score": recall_score
    }
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
