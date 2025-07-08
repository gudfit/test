# E2/tt/src/analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze_and_plot(results_df: pd.DataFrame, performance_col: str, output_dir: str):
    metric_output_dir = os.path.join(output_dir, performance_col)
    os.makedirs(metric_output_dir, exist_ok=True)
    
    if 'prompt_len_theta' in results_df.columns:
        exp_type, theta_col, theta_label = 'GPT', 'prompt_len_theta', 'Prompt Length (Tokens)'
    elif 'unmasked_ratio_theta' in results_df.columns:
        exp_type, theta_col, theta_label = 'BERT', 'unmasked_ratio_theta', 'Unmasked Ratio'
    else:
        raise ValueError("Could not determine experiment type.")
        
    print(f"\n--- Analyzing {exp_type} results using metric: '{performance_col}' ---")
    
    theta_max = results_df[theta_col].max()
    df_max_theta = results_df[results_df[theta_col] == theta_max]
    capacity = df_max_theta.groupby('model_name')[performance_col].sum().reset_index()
    print(f"Capacity Score (|C(λ, θ_max)|) based on '{performance_col}':")
    print(capacity)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=capacity, x='model_name', y=performance_col)
    plt.title(f'Performance ({performance_col}) vs. Model (Storage Budget λ)')
    plt.ylabel('Performance Score (Count of Successes)')
    plot_path = os.path.join(metric_output_dir, 'performance_vs_storage.png')
    plt.savefig(plot_path)
    plt.close()

    target_model_name = results_df['model_name'].iloc[0]
    df_target_model = results_df[results_df['model_name'] == target_model_name]
    retrieval_performance = df_target_model.groupby(theta_col)[performance_col].sum().reset_index()
    print(f"\nRetrieval Performance for {target_model_name} based on '{performance_col}':")
    print(retrieval_performance)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=retrieval_performance, x=theta_col, y=performance_col, marker='o')
    plt.title(f'Performance ({performance_col}) vs. Retrieval Budget θ for {target_model_name}')
    plt.xlabel(theta_label)
    plot_path = os.path.join(metric_output_dir, 'retrieval_degradation.png')
    plt.savefig(plot_path)
    plt.close()

    cost_performance = results_df.groupby(['model_name', theta_col]).agg(
        performance=(performance_col, 'sum'),
        avg_retrieval_cost_ms=('retrieval_cost_ms', 'mean')
    ).reset_index()
    print(f"\nCost vs. Performance Summary based on '{performance_col}':")
    print(cost_performance)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=cost_performance, x='avg_retrieval_cost_ms', y='performance', hue='model_name', style=theta_col, s=150)
    plt.title(f'Performance ({performance_col}) vs. Retrieval Cost (Latency)')
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(metric_output_dir, 'pareto_frontier.png')
    plt.savefig(plot_path)
    plt.close()
