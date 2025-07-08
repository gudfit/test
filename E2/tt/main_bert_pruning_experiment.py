# E2/tt/main_bert_pruning_experiment.py
import yaml
import pandas as pd
import os
from tqdm import tqdm
import torch

from src.data_handler import get_sentences_from_dataset
from src.model_wrapper import restore_masked_sentence
from src.metrics import calculate_semantic_similarity
from src.analysis import analyze_and_plot
from transformers import AutoModelForMaskedLM, AutoTokenizer

SEMANTIC_THRESHOLD = 0.95

def run_bert_pruning_experiment(config: dict, sentences: list[str]) -> pd.DataFrame:
    """
    Runs the experimental loop on a set of pre-pruned BERT models.
    """
    results = []
    device = config['device'] if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config['base_model_id'])
    for lambda_budget in config['lambda_budgets']:
        model_name = lambda_budget['name']
        model_path = lambda_budget['path']
        
        print(f"\nLoading pruned model: {model_name} from {model_path}")
        model = AutoModelForMaskedLM.from_pretrained(config['base_model_id'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        storage_cost = os.path.getsize(model_path)

        for sentence in tqdm(sentences, desc=f"Processing sentences for {model_name}"):
            for theta_budget in config['theta_budgets']:
                
                restored_text, latency = restore_masked_sentence(
                    model, tokenizer, sentence, theta_budget, device
                )
                
                original_clean = tokenizer.decode(tokenizer.encode(sentence, add_special_tokens=False))
                is_perfect = (restored_text.strip() == original_clean.strip())
                semantic_sim = calculate_semantic_similarity(original_clean, restored_text)
                is_semantically_eq = (semantic_sim >= SEMANTIC_THRESHOLD)

                results.append({
                    'model_name': model_name,
                    'unmasked_ratio_theta': theta_budget, 
                    'storage_cost_lambda': storage_cost,
                    'is_perfect': is_perfect,
                    'is_semantically_equivalent': is_semantically_eq,
                    'semantic_similarity': semantic_sim,
                    'retrieval_cost_ms': latency
                })
        
        del model
        torch.cuda.empty_cache()

    return pd.DataFrame(results)

def main():
    """ Main function to run the BERT pruning experiment pipeline. """
    config_path = 'configs/bert_pruning_experiment_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("--- BERT Pruning Experiment Configuration Loaded ---")
    
    print("\n--- Loading Data ---")
    sentences = get_sentences_from_dataset(config)
    print("\n--- Starting BERT Pruning Experiment ---")
    results_df = run_bert_pruning_experiment(config, sentences)
    
    print("\n--- Experiment Finished. Saving Results ---")
    output_file = config['output_file']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    print("\n--- Analyzing All Metrics from BERT Pruning Experiment ---")
    output_dir_base = os.path.join("results", "analysis_bert_pruning")
    
    analyze_and_plot(results_df, 'is_perfect', output_dir_base)
    analyze_and_plot(results_df, 'is_semantically_equivalent', output_dir_base)
    
    print(f"\n--- BERT Pruning Analysis Complete. Plots saved in '{output_dir_base}' ---")

if __name__ == "__main__":
    main()
