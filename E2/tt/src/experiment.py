# E2/tt/src/experiment.py
import pandas as pd
from tqdm import tqdm
import torch

from .model_wrapper import load_model_and_tokenizer, reconstruct_sentence

def run_experiment(config: dict, sentences: list[str]) -> pd.DataFrame:
    results = []
    device = config['device'] if torch.cuda.is_available() else "cpu"
    
    for lambda_budget in config['lambda_budgets']:
        model_name = lambda_budget['name']
        model_id = lambda_budget['model_id']
        storage_cost = lambda_budget['storage_cost_params']
        
        model, tokenizer = load_model_and_tokenizer(model_id, device)

        for sentence in tqdm(sentences, desc=f"Processing sentences for {model_name}"):
            for theta_budget in config['theta_budgets']:
                
                reconstructed_text, latency = reconstruct_sentence(
                    model, tokenizer, sentence, theta_budget, device
                )
                
                is_perfect_reconstruction = (reconstructed_text == sentence)

                results.append({
                    'model_name': model_name,
                    'prompt_len_theta': theta_budget,
                    'storage_cost_lambda': storage_cost,
                    'original_sentence': sentence,
                    'reconstructed_sentence': reconstructed_text,
                    'is_perfect': is_perfect_reconstruction,
                    'retrieval_cost_ms': latency
                })
        
        del model
        del tokenizer
        torch.cuda.empty_cache()

    return pd.DataFrame(results)

if __name__ == "__main__":
    print("--- Running experiment.py self-test ---")
    
    mock_config = {
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'lambda_budgets': [{
            'name': 'GPT2-tiny-test',
            'model_id': 'gpt2',
            'storage_cost_params': 124439808
        }],
        'theta_budgets': [1, 5]
    }
    mock_sentences = [
        "This is a test sentence for the experiment module.",
        "Another sentence to ensure the loop works correctly."
    ]
    
    df_results = run_experiment(mock_config, mock_sentences)
    
    print("\nSelf-test completed. Generated DataFrame:")
    print(df_results.head())
    assert len(df_results) == 4 
    assert 'is_perfect' in df_results.columns
    print("\nDataFrame structure is correct.")
