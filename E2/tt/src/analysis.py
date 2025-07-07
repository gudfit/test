# src/analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze_results(results_df: pd.DataFrame, config: dict):
    """
    Analyzes the results DataFrame to compute summary statistics and generate plots.
    
    Args:
        results_df (pd.DataFrame): The dataframe from the experiment.
        config (dict): The experiment configuration.
    """
    results_dir = os.path.dirname(config['output_file'])
    os.makedirs(results_dir, exist_ok=True)
    
    # --- Part A: Measuring Storage Degradation (Memorization Capacity) ---
    print("\n--- Analyzing Storage Degradation ---")
    
    # Filter results for the max theta budget to calculate C(λ, θ_max)
    theta_max = config['theta_max']
    df_max_theta = results_df[results_df['prompt_len_theta'] == theta_max]

    # Calculate |C(λ, θ_max)| for each model
    # This is the number of perfectly reconstructed sentences
    memorization_capacity = df_max_theta.groupby('model_name')['is_perfect'].sum().reset_index()
    memorization_capacity.rename(columns={'is_perfect': 'memorization_capacity'}, inplace=True)
    
    # Merge with storage cost
    storage_costs = pd.DataFrame(config['lambda_budgets'])[['name', 'storage_cost_params']].rename(columns={'name': 'model_name'})
    memorization_capacity = pd.merge(memorization_capacity, storage_costs, on='model_name')
    
    print("Memorization Capacity (|C(λ, θ_max)|):")
    print(memorization_capacity)
    
    # Plot: Performance vs. Storage Cost
    plt.figure(figsize=(10, 6))
    sns.barplot(data=memorization_capacity, x='model_name', y='memorization_capacity')
    plt.title('Performance (Memorization Capacity) vs. Model (Storage Budget λ)')
    plt.xlabel('Model')
    plt.ylabel('Number of Perfectly Reconstructed Sentences')
    plot_path = os.path.join(results_dir, 'performance_vs_storage_cost.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()

    # --- Part B: Measuring Retrieval Degradation (Contextual Efficiency) ---
    print("\n--- Analyzing Retrieval Degradation ---")
    
    # Focus on one model, e.g., the first one in the config
    target_model_name = config['lambda_budgets'][0]['name']
    df_target_model = results_df[results_df['model_name'] == target_model_name]
    
    # Calculate performance for each retrieval budget theta
    retrieval_performance = df_target_model.groupby('prompt_len_theta')['is_perfect'].sum().reset_index()
    retrieval_performance.rename(columns={'is_perfect': 'num_perfect_reconstructions'}, inplace=True)
    
    print(f"Retrieval Performance for {target_model_name}:")
    print(retrieval_performance)

    # Plot: Performance vs. Retrieval Budget
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=retrieval_performance, x='prompt_len_theta', y='num_perfect_reconstructions', marker='o')
    plt.title(f'Performance vs. Retrieval Budget θ for {target_model_name}')
    plt.xlabel('Prompt Length (Tokens)')
    plt.ylabel('Number of Perfectly Reconstructed Sentences')
    plot_path = os.path.join(results_dir, 'performance_vs_retrieval_budget.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()
    
    # --- Part C: Performance vs. Retrieval Cost (Pareto Frontier) ---
    print("\n--- Analyzing Performance vs. Retrieval Cost ---")
    
    # Calculate average latency for each (λ, θ) configuration
    cost_performance = results_df.groupby(['model_name', 'prompt_len_theta']).agg(
        performance=('is_perfect', 'sum'),
        avg_retrieval_cost_ms=('retrieval_cost_ms', 'mean')
    ).reset_index()

    print("Cost vs. Performance Summary:")
    print(cost_performance)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=cost_performance, x='avg_retrieval_cost_ms', y='performance', hue='model_name', style='prompt_len_theta', s=150)
    plt.title('Performance vs. Retrieval Cost (Latency)')
    plt.xlabel('Average Retrieval Cost (ms/sentence)')
    plt.ylabel('Number of Perfectly Reconstructed Sentences')
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'pareto_frontier.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()


# --- Self-Testing Block ---
if __name__ == '__main__':
    print("--- Running analysis.py self-test ---")
    # Create a mock dataframe and config to test the analysis functions
    mock_data = {
        'model_name': ['GPT2-small', 'GPT2-small', 'GPT2-medium', 'GPT2-medium'],
        'prompt_len_theta': [10, 5, 10, 5],
        'storage_cost_lambda': [124, 124, 355, 355],
        'is_perfect': [True, False, True, True],
        'retrieval_cost_ms': [20.5, 15.1, 45.3, 38.9]
    }
    mock_df = pd.DataFrame(mock_data)
    mock_config = {
        'output_file': 'results/test_results.csv',
        'theta_max': 10,
        'lambda_budgets': [
            {'name': 'GPT2-small', 'storage_cost_params': 124},
            {'name': 'GPT2-medium', 'storage_cost_params': 355}
        ]
    }
    analyze_results(mock_df, mock_config)
    print("\nAnalysis self-test completed.")
