#!/bin/bash
set -e

# --- Path Configuration (Self-Aware) ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# --- Main Configuration ---
MLM_MODELS=("bert-base-cased" "roberta-base" "distilroberta-base")
DATA_DIR="${PROJECT_ROOT}/data"
TRAIN_FILE="${DATA_DIR}/wikipedia.txt"
PROBE_SOURCE_FILE="${DATA_DIR}/largesample.txt"
PROBE_JSON_FILE="${SCRIPT_DIR}/probes.json"
LOG_FILE="${SCRIPT_DIR}/E1B_local_results.log"
CACHE_DIR="${SCRIPT_DIR}/cache"

export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
export TOKENIZERS_PARALLELISM=false

echo "--- Experiment 1B (Local Data): LLM as a Static Knowledge Compressor ---"
echo "--- Output will be stored in: ${SCRIPT_DIR} ---"
echo "=================================================================="

rm -f "$LOG_FILE"
touch "$LOG_FILE"

# --- Model-Agnostic Setup ---
echo -e "\nGenerating probes from ${PROBE_SOURCE_FILE}..." | tee -a "$LOG_FILE"
python -m E1.E1B.generate_probes --input-file "$PROBE_SOURCE_FILE" --output-file "$PROBE_JSON_FILE"

echo -e "\nCalculating size of Gzip-compressed data: ${TRAIN_FILE}..." | tee -a "$LOG_FILE"
GZIP_SIZE_MB=$(python -c "from E1.E1B.knowledge_quant_utils import get_gzip_size_and_compress; print(get_gzip_size_and_compress('${TRAIN_FILE}'))")
echo "Gzip Compressed Data Size (${TRAIN_FILE}): ${GZIP_SIZE_MB} MB" | tee -a "$LOG_FILE"

# --- Model-Specific Loop ---
for model_name in "${MLM_MODELS[@]}"; do
    clean_model_name=$(echo "$model_name" | tr '/' '-')
    LOCAL_MODEL_PATH="${SCRIPT_DIR}/models/${clean_model_name}-finetuned-local"

    echo -e "\n\n================== Processing Model: ${model_name} ==================" | tee -a "$LOG_FILE"

    # --- Phase 0: Setup and Fine-Tuning ---
    echo -e "\nPHASE 0: Fine-Tuning..." | tee -a "$LOG_FILE"
    if [ ! -d "$LOCAL_MODEL_PATH" ]; then
        echo "Fine-tuning ${model_name} on ${TRAIN_FILE}..." | tee -a "$LOG_FILE"
        python -m E1.E1B.setup_model_local \
            --model-name "$model_name" \
            --train-file "$TRAIN_FILE" \
            --output-path "$LOCAL_MODEL_PATH"
    else
        echo "Fine-tuned model for ${model_name} already exists. Skipping fine-tuning." | tee -a "$LOG_FILE"
    fi

    # --- Phase 1: Static Size vs. Functional Utility ---
    echo -e "\nPHASE 1: Static Size vs. Functional Utility..." | tee -a "$LOG_FILE"
    echo "Assessing Factual Recall for fine-tuned ${model_name}..." | tee -a "$LOG_FILE"
    python -m E1.E1B.eval_factual_recall \
        --model-path "$LOCAL_MODEL_PATH" \
        --probe-file "$PROBE_JSON_FILE" | tee -a "$LOG_FILE"

    # --- Phase 2: Computational Cost & Adaptability ---
    echo -e "\nPHASE 2: Computational Cost & Adaptability..." | tee -a "$LOG_FILE"
    echo "NOTE: The following tasks use the base '${model_name}' model for fair comparison." | tee -a "$LOG_FILE"

    # THIS IS THE NEW STEP
    echo -e "\nAssessing Computational Cost for ${model_name}..." | tee -a "$LOG_FILE"
    python -m E1.E1B.eval_computational_cost --model-name "$model_name" --cache-dir "$CACHE_DIR" | tee -a "$LOG_FILE"

    echo -e "\nAssessing Zero-Shot Reasoning for ${model_name}..." | tee -a "$LOG_FILE"
    python -m E1.E1B.eval_reasoning --model-name "$model_name" --cache-dir "$CACHE_DIR" | tee -a "$LOG_FILE"

    echo -e "\nMeasuring Data Efficiency for ${model_name}..." | tee -a "$LOG_FILE"
    python -m E1.E1B.eval_data_efficiency --model-name "$model_name" --cache-dir "$CACHE_DIR" | tee -a "$LOG_FILE"
done

echo "==================================================================" | tee -a "$LOG_FILE"
echo "Experiment 1B complete for all models. Results logged to ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "=================================================================="
