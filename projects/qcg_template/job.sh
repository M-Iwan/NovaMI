#!/bin/bash
set -euox pipefail

echo "Job started"
# Update paths
PYTHON_PATH="${HOME}/miniforge3/envs/pmd/bin/python"
SCRIPT_PATH="${PWD}/optim.py"

echo "Parsing arguments"
# --- Positional arguments ---
DATA_PATH="${1}"
DESC_PATH="${2}"
OUTPUT_DIR="${3}"
MODEL_NAME="${4}"
#
SMILES_COL="${5}"
FEATURES_COL="${6}"
TARGET_COL="${7}"
FOLD_COL="${8}"
WEIGHTS_COL="${9:-}"
GROUPS_COL="${10:-}"
#
OPTIM_METRIC="${11}"
N_TRIALS="${12}"
N_JOBS="${13}"
TEST_FOLD="${14}"
TASK="${15}"

# Checking packages
echo "Checking required packages"
"$PYTHON_PATH" -c "import sklearn, xgboost, optuna, numpy, pandas, polars, scipy; print('Packages OK')"

echo "Using Python at ${PYTHON_PATH}"
echo "Executing script at ${SCRIPT_PATH}"

OMP_NUM_THREADS="${N_JOBS}" MKL_NUM_THREADS="${N_JOBS}" NUMEXPR_NUM_THREADS="${N_JOBS}" OPENBLAS_NUM_THREADS="${N_JOBS}" VECLIB_MAXIMUM_THREADS="${N_JOBS}" "${PYTHON_PATH}" -u "${SCRIPT_PATH}" \
    --data_path "${DATA_PATH}" --desc_path "${DESC_PATH}" --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" --smiles_col "${SMILES_COL}" --features_col "${FEATURES_COL}" \
    --target_col "${TARGET_COL}" --fold_col "${FOLD_COL}" --weights_col "${WEIGHTS_COL}" \
    --groups_col "${GROUPS_COL}" --optim_metric "${OPTIM_METRIC}" --n_trials "${N_TRIALS}" \
    --n_jobs "${N_JOBS}" --test_fold "${TEST_FOLD}" --task "${TASK}"

echo "Job completed"