#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/../.."
E1B_DIR="${PROJECT_ROOT}/E1/E1B"

ALL_MODELS=("bert-base-cased" "roberta-base" "distilroberta-base")
PRUNE_LEVELS=(0.10 0.25 0.50)

LOG_FILE="${SCRIPT_DIR}/E1C_results.log"
CACHE_DIR="${SCRIPT_DIR}/cache"

export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
export TOKENIZERS_PARALLELISM=false

echo "--- Experiment 1C: Effects of Pruning on Downstream Performance ---" | tee -a "$LOG_FILE"
echo "===================================================================" | tee -a "$LOG_FILE"

run_evaluations() {
    local model_path="$1"
    local model_desc="$2"

    echo -e "\n--- Evaluating Model: ${model_desc} ---" | tee -a "$LOG_FILE"
    echo "Model Path: ${model_path}" | tee -a "$LOG_FILE"

    echo -e "\nRunning GLUE Benchmark Analysis..." | tee -a "$LOG_FILE"
    python -m E1.E1C.eval_glue_tasks --model-path "$model_path" --cache-dir "$CACHE_DIR" | tee -a "$LOG_FILE"

    echo -e "\nRunning Crease Magnitude Analysis..." | tee -a "$LOG_FILE"
    python -m E1.E1B.eval_crease_magnitude --compression-model "$model_path" --oracle-model "gpt2" --cache-dir "$CACHE_DIR" | tee -a "$LOG_FILE"

    echo -e "\nRunning Data Efficiency Analysis..." | tee -a "$LOG_FILE"
    python -m E1.E1B.eval_data_efficiency --model-name "$model_path" --cache-dir "$CACHE_DIR" | tee -a "$LOG_FILE"

    echo -e "\nRunning Compute and Memory Analysis..." | tee -a "$LOG_FILE"
    python -m E1.E1C.eval_compute_and_memory --model-path "$model_path" --cache-dir "$CACHE_DIR" | tee -a "$LOG_FILE"
}

for model_name in "${ALL_MODELS[@]}"; do
    clean_model_name=$(echo "$model_name" | tr '/' '-')
    BASE_MODEL_PATH="${E1B_DIR}/models/${clean_model_name}-finetuned-local"
    
    echo -e "\n\n================== Processing Model: ${model_name} ==================" | tee -a "$LOG_FILE"
    
    if [ ! -d "$BASE_MODEL_PATH" ]; then
        python -m E1.E1B.setup_model_local --model-name "$model_name" --train-file "${PROJECT_ROOT}/data/wikipedia.txt" --output-path "$BASE_MODEL_PATH"
    fi

    run_evaluations "$BASE_MODEL_PATH" "${model_name} Baseline (FP32)"

    for level in "${PRUNE_LEVELS[@]}"; do
        pruned_model_path="${SCRIPT_DIR}/models/magnitude_pruned_${clean_model_name}_$(echo $level | tr '.' p)"
        if [ ! -d "$pruned_model_path" ]; then
            python -m E1.E1C.model_compressor --model-path "$BASE_MODEL_PATH" --output-path "$pruned_model_path" --sparsity "$level"
        fi
        
        percent_level=$(awk -v level="$level" 'BEGIN { print level * 100 }')
        run_evaluations "$pruned_model_path" "${model_name} Magnitude Pruned at ${percent_level}%"
    done
done

echo -e "\n\n===================================================================" | tee -a "$LOG_FILE"
echo "Experiment 1C complete. Results logged to ${LOG_FILE}" | tee -a "$LOG_FILE"
