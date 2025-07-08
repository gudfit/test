# src/analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse  # New import for command-line arguments

def analyze_results(results_df: pd.DataFrame, performance_col: str, output_dir: str):
    """
    Analyzes the results DataFrame to compute summary statistics and generate plots.
    This version is generic and works with any specified performance column.
    
    Args:
        results_df (pd.DataFrame): The dataframe from an experiment.
        performance_col (str): The name of the column to use as the performance metric 
                               (e.g., 'is_perfect', 'is_semantically_equivalent').
        output_dir (str): The directory to save plots into.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Detect Experiment Type (GPT vs BERT) ---
    if 'prompt_len_theta' in results_df.columns:
        experiment_type = 'GPT'
        theta_col = 'prompt_len_theta'
        theta_label = 'Prompt Length (Tokens)'
    elif 'unmasked_ratio_theta' in results_df.columns:
        experiment_type = 'BERT'
        theta_col = 'unmasked_ratio_theta'
        theta_label = 'Unmasked Ratio'
    else:
        raise ValueError("Could not determine experiment type from DataFrame columns.")
        
    print(f"\n--- Analyzing {experiment_type} results using metric: '{performance_col}' ---")
    
    # --- Part A: Measuring Storage Degradation (Memorization Capacity) ---
    print("\n--- Analyzing Storage Degradation ---")
    
    # Get the max theta value from the data itself
    theta_max = results_df[theta_col].max()
    df_max_theta = results_df[results_df[theta_col] == theta_max]

    # Calculate capacity based on the specified performance column
    capacity = df_max_theta.groupby('model_name')[performance_col].sum().reset_index()
    capacity.rename(columns={performance_col: 'capacity_score'}, inplace=True)
    
    # Get storage cost from the dataframe
    storage_costs = results_df[['model_name', 'storage_cost_lambda']].drop_duplicates()
    capacity = pd.merge(capacity, storage_costs, on='model_name')
    
    print("Capacity Score (|C(λ, θ_max)|):")
    print(capacity)
    
    # Plot: Performance vs. Storage Cost
    plt.figure(figsize=(10, 6))
    sns.barplot(data=capacity, x='model_name', y='capacity_score')
    plt.title(f'Performance ({performance_col}) vs. Model (Storage Budget λ)')
    plt.xlabel('Model')
    plt.ylabel('Performance Score (Count of Successes)')
    plot_path = os.path.join(output_dir, f'performance_vs_storage_{performance_col}.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()

    # --- Part B: Measuring Retrieval Degradation (Contextual Efficiency) ---
    print("\n--- Analyzing Retrieval Degradation ---")
    
    target_model_name = results_df['model_name'].iloc[0]
    df_target_model = results_df[results_df['model_name'] == target_model_name]
    
    retrieval_performance = df_target_model.groupby(theta_col)[performance_col].sum().reset_index()
    
    print(f"Retrieval Performance for {target_model_name}:")
    print(retrieval_performance)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=retrieval_performance, x=theta_col, y=performance_col, marker='o')
    plt.title(f'Performance ({performance_col}) vs. Retrieval Budget θ for {target_model_name}')
    plt.xlabel(theta_label)
    plt.ylabel('Performance Score (Count of Successes)')
    plot_path = os.path.join(output_dir, f'retrieval_degradation_{performance_col}.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()
    
    # --- Part C: Performance vs. Retrieval Cost (Pareto Frontier) ---
    print("\n--- Analyzing Performance vs. Retrieval Cost ---")
    
    cost_performance = results_df.groupby(['model_name', theta_col]).agg(
        performance=(performance_col, 'sum'),
        avg_retrieval_cost_ms=('retrieval_cost_ms', 'mean')
    ).reset_index()

    print("Cost vs. Performance Summary:")
    print(cost_performance)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=cost_performance, x='avg_retrieval_cost_ms', y='performance', hue='model_name', style=theta_col, s=150)
    plt.title(f'Performance ({performance_col}) vs. Retrieval Cost (Latency)')
    plt.xlabel('Average Retrieval Cost (ms/instance)')
    plt.ylabel('Performance Score (Count of Successes)')
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'pareto_frontier_{performance_col}.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()


def main():
    """
    Main entry point for the analysis script, run from the command line.
    """
    parser = argparse.ArgumentParser(description="Analyze LLM compression experiment results.")
    
    parser.add_argument(
        "results_file", 
        type=str, 
        help="Path to the experiment results CSV file."
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        default="is_perfect", 
        choices=["is_perfect", "is_semantically_equivalent"],
        help="The performance metric column to use for analysis."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found at {args.results_file}")
        return
        
    print(f"Loading results from: {args.results_file}")
    results_df = pd.read_csv(args.results_file)
    
    if args.metric not in results_df.columns:
        print(f"Error: Metric column '{args.metric}' not found in the results file.")
        print(f"Available columns are: {results_df.columns.tolist()}")
        return

    # Create a unique output directory for this analysis run
    base_name = os.path.splitext(os.path.basename(args.results_file))[0]
    output_dir = os.path.join("results", f"analysis_{base_name}_{args.metric}")

    analyze_results(results_df, args.metric, output_dir)
    
    print(f"\nAnalysis complete. Plots saved in '{output_dir}'")

if __name__ == '__main__':
    main()
