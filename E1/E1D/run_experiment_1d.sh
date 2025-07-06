#!/bin/bash
set -e

# --- Path Configuration ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# --- Main Configuration ---
ALL_MODELS=(
    "EleutherAI/gpt-neo-2.7B"
    "gpt2"
    "facebook/opt-1.3b"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "bert-base-cased"
    "roberta-base"
    "distilroberta-base"
)
DATA_DIR="${PROJECT_ROOT}/data"
LOCAL_FILES=("largesample.txt" "othello.txt" "sample.txt" "wikipedia.txt")

CACHE_DIR="${SCRIPT_DIR}/cache"

export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
export TOKENIZERS_PARALLELISM=false

echo "--- Experiment 1D: LLM vs. Algorithmic Compression Benchmark ---"
echo "====================================================================="

echo -e "\nINFO: This script uses external compressors."
echo " - 'zstd' should be installed via your system's package manager (e.g., 'apt-get install zstd')."

for file in "${LOCAL_FILES[@]}"; do
    DATASET_FILE_PATH="${DATA_DIR}/${file}"
    
    if [ ! -f "$DATASET_FILE_PATH" ]; then
        echo "ERROR: Dataset file not found at ${DATASET_FILE_PATH}. Skipping."
        continue
    fi

    for model_name in "${ALL_MODELS[@]}"; do
        echo -e "\n\n================== Benchmarking: ${file} vs. ${model_name} =================="
        echo -e "\nStarting benchmark process... This may take a while."
        python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('${model_name}', cache_dir='${CACHE_DIR}')" > /dev/null 2>&1
        
        clean_model_name=$(echo "$model_name" | tr '/' '-')
        base_model_size_bytes=$(python -c "import sys; sys.path.append('.'); from E1.E1D.method_lossless_benchmark import get_dir_size_bytes; print(get_dir_size_bytes('${CACHE_DIR}/models--${clean_model_name}'))")

        python -m E1.E1D.method_lossless_benchmark \
            --model-name "$model_name" \
            --dataset-file "$DATASET_FILE_PATH" \
            --cache-dir "$CACHE_DIR" \
            --base-model-size "$base_model_size_bytes"
    done
done

echo -e "\nExperiment 1D complete for all local files and models."
