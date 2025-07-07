# main.py
import yaml
import pandas as pd
import os

from src.data_handler import get_sentences_from_dataset
from src.experiment import run_experiment
from src.analysis import analyze_results

def main():
    """
    Main function to run the entire pipeline:
    1. Load configuration.
    2. Load data.
    3. Run experiment.
    4. Save results.
    5. Analyze results and generate plots.
    """
    # 1. Load Configuration
    config_path = 'configs/experiment_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("--- Configuration Loaded ---")
    print(yaml.dump(config, indent=2))
    
    # 2. Load Data
    print("\n--- Loading Data ---")
    sentences = get_sentences_from_dataset(config)
    
    # For a quick run, you might want to slice the data
    # sentences = sentences[:100] # Uncomment for a quick test run
    
    # 3. Run Experiment
    print("\n--- Starting Experiment ---")
    results_df = run_experiment(config, sentences)
    
    # 4. Save Results
    print("\n--- Experiment Finished. Saving Results ---")
    output_file = config['output_file']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # 5. Analyze Results
    print("\n--- Analyzing Results ---")
    analyze_results(results_df, config)
    
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main()
