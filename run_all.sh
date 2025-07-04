#!/bin/bash

set -e
BENCHMARK_MODELS=(
    "bert-base-cased"
    "roberta-base"
)
MASK_RATIOS=(0.15 0.30 0.50)
FINAL_EPOCH=10
RESULTS_FILE="results/master_log.json"
echo "--- Step 1: Setting up Python Environment ---"
pip install -r requirements.txt
echo "Environment setup complete."
echo "================================================="
echo "--- Step 2: Verifying Local Data Files ---"
if ! [ -f "data/othello.txt" ] || ! [ -f "data/wiki.txt" ]; then
    echo "ERROR: Missing data files. Please ensure 'othello.txt' and 'wiki.txt' exist in the 'data/' directory."
    exit 1
fi
echo "Local data files found."
echo "================================================="
echo "Clearing previous results file: ${RESULTS_FILE}"
rm -f ${RESULTS_FILE}
touch ${RESULTS_FILE}
echo "--- Starting Experiment Block 1: Othello (Local File) ---"
OTHELLO_NICKNAME="Othello"
OTHELLO_FILE="data/othello.txt"
OTHELLO_MODEL="bert-base-cased"
SAFE_MODEL_NAME=$(basename "${OTHELLO_MODEL}")

echo ">>> Training ${OTHELLO_MODEL} on ${OTHELLO_NICKNAME}..."
python main.py --train \
    --model-name "${OTHELLO_MODEL}" \
    --file-path "${OTHELLO_FILE}" \
    --dataset-nickname "${OTHELLO_NICKNAME}"

OTHELLO_MODEL_PATH="models/${SAFE_MODEL_NAME}_${OTHELLO_NICKNAME}_epoch${FINAL_EPOCH}"
for ratio in "${MASK_RATIOS[@]}"; do
    echo ">>> Analyzing Othello model at mask ratio: ${ratio} <<<"
    python main.py --run-analysis \
        --model-path "${OTHELLO_MODEL_PATH}" \
        --file-path "${OTHELLO_FILE}" \
        --mask-ratio "${ratio}" \
        | tee -a ${RESULTS_FILE}
    echo "" >> ${RESULTS_FILE}
done
echo "--- Othello Experiment Finished ---"
echo "================================================="


# --- Experiment Block 2: Local File - Wiki.txt ---
echo "--- Starting Experiment Block 2: Wiki.txt (Local File) ---"
WIKI_NICKNAME="WikiLocal"
WIKI_FILE="data/wiki.txt"
WIKI_MODEL="bert-base-cased"
SAFE_MODEL_NAME=$(basename "${WIKI_MODEL}")

echo ">>> Training ${WIKI_MODEL} on ${WIKI_NICKNAME}..."
python main.py --train \
    --model-name "${WIKI_MODEL}" \
    --file-path "${WIKI_FILE}" \
    --dataset-nickname "${WIKI_NICKNAME}"

WIKI_MODEL_PATH="models/${SAFE_MODEL_NAME}_${WIKI_NICKNAME}_epoch${FINAL_EPOCH}"
for ratio in "${MASK_RATIOS[@]}"; do
    echo ">>> Analyzing Wiki.txt model at mask ratio: ${ratio} <<<"
    python main.py --run-analysis \
        --model-path "${WIKI_MODEL_PATH}" \
        --file-path "${WIKI_FILE}" \
        --mask-ratio "${ratio}" \
        | tee -a ${RESULTS_FILE}
    echo "" >> ${RESULTS_FILE}
done
echo "--- Wiki.txt Experiment Finished ---"
echo "================================================="
echo "--- Starting Experiment Block 3: WikiText-2 (HF Benchmark) ---"
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"

for model in "${BENCHMARK_MODELS[@]}"; do
    echo -e "\n>>> Training benchmark model: ${model} <<<"
    python main.py --train \
        --model-name "${model}" \
        --dataset-name "${DATASET_NAME}" \
        --dataset-config "${DATASET_CONFIG}"
    
    SAFE_MODEL_NAME=$(basename "${model}")
    MODEL_PATH="models/${SAFE_MODEL_NAME}_${DATASET_NAME}_epoch${FINAL_EPOCH}"
    
    for ratio in "${MASK_RATIOS[@]}"; do
        echo ">>> Analyzing ${model} at mask ratio: ${ratio} <<<"
        python main.py --run-analysis \
            --model-path "${MODEL_PATH}" \
            --dataset-name "${DATASET_NAME}" \
            --dataset-config "${DATASET_CONFIG}" \
            --mask-ratio "${ratio}" \
            | tee -a ${RESULTS_FILE}
        echo "" >> ${RESULTS_FILE}
    done
done
echo "--- Benchmark Experiment Finished ---"
echo "================================================="

echo "--- ALL EXPERIMENTS HAVE FINISHED SUCCESSFULLY! ---"
echo "Comprehensive results are logged in: ${RESULTS_FILE}"
