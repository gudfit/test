#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
RESULTS_DIR="${SCRIPT_DIR}/results"
MODELS_DIR="${SCRIPT_DIR}/models"
TRAINING_RESULTS_FILE="${RESULTS_DIR}/training_results.json"
BAKEOFF_RESULTS_FILE="${RESULTS_DIR}/sota_bakeoff_log.json"

MLM_MODELS=("bert-base-cased" "roberta-base" "distilroberta-base")
CAUSAL_MODEL="gpt2"
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"
MASK_RATIOS=(0.25 0.50 0.75)

mkdir -p "${RESULTS_DIR}" "${MODELS_DIR}"
rm -f "${TRAINING_RESULTS_FILE}" "${BAKEOFF_RESULTS_FILE}"
touch "${TRAINING_RESULTS_FILE}" "${BAKEOFF_RESULTS_FILE}"

export PYTHONPATH="$PYTHONPATH:${SCRIPT_DIR}/.."

uv pip install -r "${SCRIPT_DIR}/requirements.txt"

run_with_flush() {
	echo "RUNNING: $@" | tee -a "${BAKEOFF_RESULTS_FILE}"
	PYTHONPATH=. python -m "$@" 2>&1 | tee -a "${BAKEOFF_RESULTS_FILE}"
	echo "----------------------------------------" | tee -a "${BAKEOFF_RESULTS_FILE}"
	python -c "import sys; sys.stdout.flush()"
}

for model_name in "${MLM_MODELS[@]}"; do
	model_path="${MODELS_DIR}/${model_name}_${DATASET_NAME}_finetuned"

	echo "=== Training ${model_name} ===" | tee -a "${TRAINING_RESULTS_FILE}"
	PYTHONPATH=. python -m E1.E1A.setup_model \
		--model-name "${model_name}" \
		--dataset-name "${DATASET_NAME}" \
		--dataset-config "${DATASET_CONFIG}" \
		--output-path "${model_path}" \
		--results-file "${TRAINING_RESULTS_FILE}"
done

CAUSAL_MODEL_PATH="${MODELS_DIR}/${CAUSAL_MODEL}_${DATASET_NAME}_finetuned"
if [ ! -d "${CAUSAL_MODEL_PATH}" ]; then
	PYTHONPATH=. python -m E1.E1A.setup_model \
		--model-name "${CAUSAL_MODEL}" \
		--dataset-name "${DATASET_NAME}" \
		--dataset-config "${DATASET_CONFIG}" \
		--output-path "${CAUSAL_MODEL_PATH}" \
		--results-file "${TRAINING_RESULTS_FILE}"
fi

for model_name in "${MLM_MODELS[@]}"; do
	model_path="${MODELS_DIR}/${model_name}_${DATASET_NAME}_finetuned"

	for ratio in "${MASK_RATIOS[@]}"; do
		run_with_flush E1.E1A.method_predictive_masking \
			--model-path "${model_path}" \
			--dataset-name "${DATASET_NAME}" \
			--dataset-config "${DATASET_CONFIG}" \
			--mask-ratio "${ratio}" \
			--masking-type "random"
	done

	for ratio in "${MASK_RATIOS[@]}"; do
		run_with_flush E1.E1A.method_predictive_masking \
			--model-path "${model_path}" \
			--dataset-name "${DATASET_NAME}" \
			--dataset-config "${DATASET_CONFIG}" \
			--mask-ratio "${ratio}" \
			--masking-type "deterministic"
	done

	LSQ_DECODER_PATH="${MODELS_DIR}/${model_name}_lsq_decoder.pth"
	if [ ! -f "${LSQ_DECODER_PATH}" ]; then
		run_with_flush E1.E1A.method_latent_quantization \
			--model-path "${model_path}" \
			--dataset-name "${DATASET_NAME}" \
			--dataset-config "${DATASET_CONFIG}" \
			--decoder-path "${LSQ_DECODER_PATH}" \
			--train-decoder
	fi

	run_with_flush E1.E1A.method_latent_quantization \
		--model-path "${model_path}" \
		--dataset-name "${DATASET_NAME}" \
		--dataset-config "${DATASET_CONFIG}" \
		--decoder-path "${LSQ_DECODER_PATH}" \
		--evaluate
done

run_with_flush E1.E1A.method_arithmetic_coding \
	--model-path "${CAUSAL_MODEL_PATH}" \
	--dataset-name "${DATASET_NAME}" \
	--dataset-config "${DATASET_CONFIG}"

echo "--- SOTA BAKE-OFF COMPLETE! ---" | tee -a "${BAKEOFF_RESULTS_FILE}"
