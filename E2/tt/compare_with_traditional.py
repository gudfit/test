import os
import time
import pandas as pd
import yaml
import zlib
import bz2

from src.data_handler import get_sentences_from_dataset

def benchmark_traditional_compressor(algorithm_name: str, data_bytes: bytes) -> dict:
    # This function remains unchanged.
    if algorithm_name == 'gzip':
        compress_func = zlib.compress
        decompress_func = zlib.decompress
    elif algorithm_name == 'bzip2':
        compress_func = bz2.compress
        decompress_func = bz2.decompress
    else:
        raise ValueError("Unsupported algorithm")
    
    start_time = time.time()
    compressed_data = compress_func(data_bytes)
    original_size_bytes = len(data_bytes)
    compressed_size_bytes = len(compressed_data)
    
    start_time = time.time()
    decompressed_data = decompress_func(compressed_data)
    decompression_time = time.time() - start_time
    
    assert data_bytes == decompressed_data, f"{algorithm_name} failed verification!"

    return {
        'Algorithm': algorithm_name, 'Performance': "Lossless",
        'Storage Cost (Bytes)': compressed_size_bytes,
        'Retrieval Cost (Decompression Time ms)': decompression_time * 1000,
        'Compression Ratio': original_size_bytes / compressed_size_bytes,
        'Bits per Character': (compressed_size_bytes * 8) / len(data_bytes.decode('utf-8'))
    }


def summarize_llm_results(config: dict, results_df: pd.DataFrame, model_name: str, theta_col_name: str) -> dict:
    """
    Summarizes results for a SINGLE LLM. This is now a more generic helper function.
    """
    df_model = results_df[(results_df['model_name'] == model_name) & (results_df[theta_col_name] == config['theta_max'])]
    
    model_config = next(item for item in config['lambda_budgets'] if item["name"] == model_name)
    storage_cost_bytes = model_config['storage_cost_params'] * 2

    num_total_sentences = len(df_model)
    num_perfect = df_model['is_perfect'].sum()
    avg_retrieval_cost_ms = df_model['retrieval_cost_ms'].mean()
    
    performance_str = "N/A"
    if num_total_sentences > 0:
        performance_str = f"{num_perfect}/{num_total_sentences} ({num_perfect/num_total_sentences:.2%}) sentences perfectly recalled/restored"

    return {
        'Algorithm': model_name,
        'Performance': performance_str,
        'Storage Cost (Bytes)': storage_cost_bytes,
        'Retrieval Cost (Decompression Time ms)': avg_retrieval_cost_ms,
        'Nature of Loss': "Semantic and Graded (Lossy)",
        'Adaptability': "High (via Fine-Tuning)"
    }

def print_summary(summary: dict):
    """Helper function to print a formatted summary."""
    print(f"\nAlgorithm: {summary['Algorithm']}")
    print(f"  - Performance Nature:       {summary['Performance']}")
    print(f"  - Storage Cost (Model):     {summary['Storage Cost (Bytes)']:>10,} bytes (fixed overhead)")
    print(f"  - BPC (Bits Per Character): N/A (Fundamentally different paradigm)")
    print(f"  - Retrieval Cost (Infer):   {summary['Retrieval Cost (Decompression Time ms)']:>8.2f} ms per instance (on average)")
    print(f"  - Nature of Loss:           {summary['Nature of Loss']}")
    print(f"  - Adaptability:             {summary['Adaptability']}")


def main():
    """ Main function to run the full, comprehensive comparison. """
    # --- 1. Load Corpus and Benchmark Traditional Compressors ---
    with open('configs/experiment_config.yaml', 'r') as f:
        gpt_config = yaml.safe_load(f)
    sentences = get_sentences_from_dataset(gpt_config)
    full_corpus_bytes = "\n".join(sentences).encode('utf-8')
    
    traditional_results = [benchmark_traditional_compressor(algo, full_corpus_bytes) for algo in ['gzip', 'bzip2']]

    # --- 2. Load ALL LLM Results ---
    df_gpt = pd.read_csv('results/reconstruction_results.csv')
    df_bert = pd.read_csv('results/bert_restoration_results.csv')
    with open('configs/bert_experiment_config.yaml', 'r') as f:
        bert_config = yaml.safe_load(f)

    # --- 3. Generate Summaries for ALL Models ---
    llm_summaries = []
    # Summarize GPT models
    for model_budget in gpt_config['lambda_budgets']:
        summary = summarize_llm_results(gpt_config, df_gpt, model_budget['name'], 'prompt_len_theta')
        llm_summaries.append(summary)
        
    # Summarize BERT models
    for model_budget in bert_config['lambda_budgets']:
        summary = summarize_llm_results(bert_config, df_bert, model_budget['name'], 'unmasked_ratio_theta')
        llm_summaries.append(summary)

    # --- 4. Present The Grand Comparison Table ---
    print("\n\n" + "="*80)
    print("--- THE GRAND COMPARISON: LLMs vs. TRADITIONAL COMPRESSORS ---")
    print("="*80 + "\n")

    print("--- Traditional Compressors (Lossless Syntactic Compression) ---")
    for summary in traditional_results:
        print(f"\nAlgorithm: {summary['Algorithm']}")
        print(f"  - Performance Nature:       {summary['Performance']}")
        print(f"  - Storage Cost (Compressed):{summary['Storage Cost (Bytes)']:>10,} bytes ({summary['Compression Ratio']:.2f}x ratio)")
        print(f"  - BPC (Bits Per Character): {summary['Bits per Character']:.3f}")
        print(f"  - Retrieval Cost (Decompress):{summary['Retrieval Cost (Decompression Time ms)']:>8.2f} ms for the entire corpus")
    
    print("\n\n--- Large Language Models (Lossy Semantic Compression) ---")
    for summary in llm_summaries:
        print_summary(summary)
    
    print("\n\n" + "="*80)
    print("Comparison complete. The table above provides the data for Objective 3.")
    print("="*80)

if __name__ == "__main__":
    main()
