#!/bin/bash

# Finish on error
set -euox pipefail
DIR=$(pwd)

# Specify paths to Python and the
PYTHON_PATH="${HOME}/miniforge3/envs/pmd/bin/python"
# Path to Python script that's submitting individual jobs; assuming it's in the same directory
SCRIPT_PATH="${DIR}/manager.py"

# Number of available CPUs
N_CPUS=96
# Number of CPUS per task
N_JOBS=1

# Because the whole setup is somewhat complicated, we explicitly limit available CPUs using taskset
CPU_RANGE="0-$((N_CPUS-1))"

OMP_NUM_THREADS="${N_JOBS}" MKL_NUM_THREADS="${N_JOBS}" NUMEXPR_NUM_THREADS="${N_JOBS}" VECLIB_MAXIMUM_THREADS="${N_JOBS}" OPENBLAS_NUM_THREADS="${N_JOBS}" taskset -c ${CPU_RANGE} nohup "${PYTHON_PATH}" "${SCRIPT_PATH}" > run_qcg.log 2>&1 &
