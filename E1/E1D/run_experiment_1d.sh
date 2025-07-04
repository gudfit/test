#!/bin/bash
set -e

DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-103-v1"
LLM_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE="wikitext-103-test.txt"
SAFE_MODEL_NAME=$(basename "${LLM_MODEL}")
FINE_TUNED_MODEL_PATH="E1/E1D/models/${SAFE_MODEL_NAME}_${DATASET_NAME}_finetuned"

echo "--- Experiment 1D: Comparative Analysis (using ${LLM_MODEL}) ---"

echo -e "\n--- Step 1: Preparing test data and prerequisites ---"
for cmd in gzip bzip2 zstd bc; do
  if ! command -v $cmd &> /dev/null; then
    echo "ERROR: Command '$cmd' not found. Please install it."
    exit 1
  fi
done

if [ ! -f "${DATA_FILE}" ]; then
    echo "Downloading and concatenating dataset to ${DATA_FILE}..."
    python -c "from datasets import load_dataset; ds = load_dataset('${DATASET_NAME}', '${DATASET_CONFIG}', split='test'); f = open('${DATA_FILE}', 'w'); f.write('\n'.join(x for x in ds['text'] if x)); f.close()"
else
    echo "${DATA_FILE} already exists. Skipping download."
fi
ORIGINAL_SIZE_BYTES=$(wc -c < "${DATA_FILE}")
echo "Test data prepared. Original size: ${ORIGINAL_SIZE_BYTES} bytes."

echo -e "\n--- Step 2: Fine-tuning ${LLM_MODEL} on ${DATASET_NAME} ---"
echo "NOTE: Fine-tuning a 1.1B parameter model may take some time and VRAM."
if [ ! -d "${FINE_TUNED_MODEL_PATH}" ]; then
    
    PYTHONPATH=. python -m E1.E1A.setup_model \
        --model-name "${LLM_MODEL}" \
        --dataset-name "${DATASET_NAME}" \
        --dataset-config "${DATASET_CONFIG}" \
        --output-path "${FINE_TUNED_MODEL_PATH}"
else
    echo "Fine-tuned model already exists at ${FINE_TUNED_MODEL_PATH}. Skipping fine-tuning."
fi

echo -e "\n--- Step 3: Benchmarking Traditional Algorithms ---"
declare -A results


echo "Running gzip..."
gzip -k -f -9 "${DATA_FILE}"
COMPRESSED_SIZE_BYTES=$(wc -c < "${DATA_FILE}.gz")
BPC=$(echo "scale=4; ($COMPRESSED_SIZE_BYTES * 8) / $ORIGINAL_SIZE_BYTES" | bc)
results["gzip -9"]=$BPC
echo "gzip -9: ${BPC} BPC"


echo "Running bzip2..."
bzip2 -k -f -9 "${DATA_FILE}"
COMPRESSED_SIZE_BYTES=$(wc -c < "${DATA_FILE}.bz2")
BPC=$(echo "scale=4; ($COMPRESSED_SIZE_BYTES * 8) / $ORIGINAL_SIZE_BYTES" | bc)
results["bzip2 -9"]=$BPC
echo "bzip2 -9: ${BPC} BPC"


echo "Running zstd..."
zstd -k -f -19 "${DATA_FILE}"
COMPRESSED_SIZE_BYTES=$(wc -c < "${DATA_FILE}.zst")
BPC=$(echo "scale=4; ($COMPRESSED_SIZE_BYTES * 8) / $ORIGINAL_SIZE_BYTES" | bc)
results["zstd -19"]=$BPC
echo "zstd -19: ${BPC} BPC"


echo -e "\n--- Step 4: Benchmarking Fine-Tuned LLM (${LLM_MODEL}) ---"

LLM_RESULT_LINE=$(PYTHONPATH=. python -m E1.E1D.method_lossless_benchmark \
    --model-name "${FINE_TUNED_MODEL_PATH}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-config "${DATASET_CONFIG}")

LLM_BPC=$(echo "$LLM_RESULT_LINE" | grep -oP '(\d+\.\d+)(?= BPC)')
results["Fine-tuned ${SAFE_MODEL_NAME}"]=$LLM_BPC
echo "LLM Result: ${LLM_BPC} BPC"


echo -e "\n--- Final Results Summary (Bits Per Character) ---"
echo "=================================================="
printf "%-45s | %-10s\n" "Compressor" "BPC"
echo "----------------------------------------------------------------"

sorted_keys=($(
    for k in "${!results[@]}"; do
        echo "$k"
    done | sort
))
for compressor in "${sorted_keys[@]}"; do
  printf "%-45s | %-10s\n" "$compressor" "${results[$compressor]}"
done
echo "=================================================="
