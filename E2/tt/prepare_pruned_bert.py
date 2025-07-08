# E2/tt/prepare_pruned_bert.py
import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForMaskedLM
import os
import copy

BASE_MODEL_ID = "bert-base-uncased"
PRUNING_AMOUNTS = [0.2, 0.4, 0.6, 0.8] 
SAVE_DIR = "models/pruned_bert_base/"

def main():
    """
    Loads a base BERT model, prunes it to different levels, and saves the state dicts.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Loading base model: {BASE_MODEL_ID}")
    base_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_ID)
    base_model.eval()

    parameters_to_prune = []
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    original_save_path = os.path.join(SAVE_DIR, "pruned_0.pt")
    torch.save(base_model.state_dict(), original_save_path)
    print(f"Saved original (0% pruned) model to {original_save_path}")

    for amount in PRUNING_AMOUNTS:
        print(f"\nCreating model with {amount*100:.0f}% pruning...")
        
        model_to_prune = copy.deepcopy(base_model)
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

        for module, param_name in parameters_to_prune:
            if prune.is_pruned(module):
                prune.remove(module, param_name)
        
        pruned_save_path = os.path.join(SAVE_DIR, f"pruned_{int(amount*100)}.pt")
        torch.save(model_to_prune.state_dict(), pruned_save_path)
        print(f"Saved {amount*100:.0f}% pruned model to {pruned_save_path}")

if __name__ == "__main__":
    main()
