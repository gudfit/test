# E2/tt/main_bert.py
import yaml
import pandas as pd
import os
from tqdm import tqdm
import torch

from src.data_handler import get_sentences_from_dataset
from src.model_wrapper import load_mlm_model_and_tokenizer, restore_masked_sentence
from src.metrics import calculate_semantic_similarity
from src.analysis import analyze_and_plot 

SEMANTIC_THRESHOLD = 0.95

def run_bert_experiment_v2(config: dict, sentences: list[str]) -> pd.DataFrame:
    """
    This function is unchanged. It runs the experiment and returns the full dataframe.
    """
    results = []
    device = config['device'] if torch.cuda.is_available() else "cpu"
    
    for lambda_budget in config['lambda_budgets']:
        model_name = lambda_budget['name']
        model_id = lambda_budget['model_id']
        storage_cost = lambda_budget['storage_cost_params']
        
        model, tokenizer = load_mlm_model_and_tokenizer(model_id, device)

        for sentence in tqdm(sentences, desc=f"Processing sentences for {model_name}"):
            original_clean = tokenizer.decode(tokenizer.encode(sentence, add_special_tokens=False))

            for theta_budget in config['theta_budgets']:
                restored_text, latency = restore_masked_sentence(
                    model, tokenizer, sentence, theta_budget, device
                )
                
                is_perfect_restoration = (restored_text.strip() == original_clean.strip())
                semantic_sim = calculate_semantic_similarity(original_clean, restored_text)
                is_semantically_eq = (semantic_sim >= SEMANTIC_THRESHOLD)
                
                results.append({
                    'model_name': model_name,
                    'unmasked_ratio_theta': theta_budget,
                    'storage_cost_lambda': storage_cost,
                    'original_sentence': sentence,
                    'restored_sentence': restored_text,
                    'is_perfect': is_perfect_restoration,
                    'is_semantically_equivalent': is_semantically_eq,
                    'semantic_similarity': semantic_sim,
                    'retrieval_cost_ms': latency
                })
        
        del model, tokenizer
        torch.cuda.empty_cache()

    return pd.DataFrame(results)

def main_bert_v2():
    """ 
    Main function to run the entire enhanced BERT pipeline and analyze BOTH metrics. 
    """
    config_path = 'configs/bert_experiment_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_filename = 'bert_semantic_results.csv'
    config['output_file'] = os.path.join('results', output_filename)
    
    print("--- BERT Semantic Experiment Configuration Loaded ---")
    print(yaml.dump(config, indent=2))
    
    print("\n--- Loading Data ---")
    sentences = get_sentences_from_dataset(config)
    print("\n--- Starting BERT Semantic Experiment ---")
    results_df = run_bert_experiment_v2(config, sentences)
    
    print("\n--- Experiment Finished. Saving Results ---")
    os.makedirs(os.path.dirname(config['output_file']), exist_ok=True)
    results_df.to_csv(config['output_file'], index=False)
    print(f"Results saved to {config['output_file']}")
    
    print("\n--- Analyzing All Metrics from Experiment Run ---")
    output_dir_base = os.path.join("results", "analysis_bert_semantic")
    analyze_and_plot(results_df, 'is_perfect', output_dir_base)
    analyze_and_plot(results_df, 'is_semantically_equivalent', output_dir_base)
    print(f"\n--- Full Analysis Complete. Plots saved in '{output_dir_base}' ---")

if __name__ == "__main__":
    main_bert_v2()
