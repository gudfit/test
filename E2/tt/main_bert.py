# main_bert.py
import yaml
import pandas as pd
import os
from tqdm import tqdm
import torch

# We can reuse the data handler
from src.data_handler import get_sentences_from_dataset
# We import the new MLM functions from our extended wrapper
from src.model_wrapper import load_mlm_model_and_tokenizer, restore_masked_sentence
# We can reuse the analysis module, it's flexible enough
from src.analysis import analyze_results

def run_bert_experiment(config: dict, sentences: list[str]) -> pd.DataFrame:
    """
    Runs the full experimental loop for Masked Language Models.
    """
    results = []
    device = config['device'] if torch.cuda.is_available() else "cpu"
    
    for lambda_budget in config['lambda_budgets']:
        model_name = lambda_budget['name']
        model_id = lambda_budget['model_id']
        storage_cost = lambda_budget['storage_cost_params']
        
        model, tokenizer = load_mlm_model_and_tokenizer(model_id, device)

        for sentence in tqdm(sentences, desc=f"Processing sentences for {model_name}"):
            # Clean original sentence for fair comparison, as BERT tokenizers might add spaces
            original_clean = tokenizer.decode(tokenizer.encode(sentence, add_special_tokens=False))

            for theta_budget in config['theta_budgets']:
                restored_text, latency = restore_masked_sentence(
                    model, tokenizer, sentence, theta_budget, device
                )
                
                # Clean restored text to remove artifacts from tokenization
                restored_clean = tokenizer.decode(tokenizer.encode(restored_text, add_special_tokens=False))
                
                is_perfect_restoration = (restored_clean == original_clean)

                results.append({
                    'model_name': model_name,
                    'unmasked_ratio_theta': theta_budget,
                    'storage_cost_lambda': storage_cost,
                    'original_sentence': sentence,
                    'restored_sentence': restored_text,
                    'is_perfect': is_perfect_restoration,
                    'retrieval_cost_ms': latency
                })
        
        del model
        del tokenizer
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


def main_bert():
    """ Main function to run the entire BERT pipeline. """
    config_path = 'configs/bert_experiment_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("--- BERT Experiment Configuration Loaded ---")
    print(yaml.dump(config, indent=2))
    
    print("\n--- Loading Data ---")
    sentences = get_sentences_from_dataset(config)
    # sentences = sentences[:100] # Uncomment for a quick test run
    
    print("\n--- Starting BERT Experiment ---")
    results_df = run_bert_experiment(config, sentences)
    
    print("\n--- Experiment Finished. Saving Results ---")
    output_file = config['output_file']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    print("\n--- Analyzing BERT Results ---")
    # We need a small tweak for analysis: rename the theta column for the plots
    results_df.rename(columns={'unmasked_ratio_theta': 'prompt_len_theta'}, inplace=True)
    analyze_results(results_df, config)
    
    print("\n--- BERT Pipeline Complete ---")

if __name__ == "__main__":
    main_bert()
