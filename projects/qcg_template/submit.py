import os
import time
import datetime
from itertools import product

from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager

def log(message: str):
    log_time = str(datetime.datetime.now()).split('.')[0]
    print(f'{log_time}: {message}')

user = 'gptil'

# Python and job paths
python_path = f"/home/{user}/miniforge3/envs/pmd/bin/python"
script_path = f"/home/{user}/Repos/PMD/src/QCG/job.sh"

# Shared arguments
data_path="file.joblib"
desc_path="file.joblib"
output_dir="/"
smiles_col="SMILES"
target_col="pIC50"
fold_col="Fold"
weights_col="Weights"  # "" if not used
groups_col="Group"  # "" if not used
optim_metric="RMSE"
n_trials=64
task="regression"
os.makedirs(output_dir, exist_ok=True)

# QCG params
max_concurrent_jobs = 96 # should correspond to n_cpus/n_jobs
n_jobs=1  # number of CPUs per job, passed directly to models
sleep_interval = 300  # check for new jobs every 5 minutes

# Define loops
model_names = []
feature_names = []
test_folds = []

# Set up Local manager
manager = LocalManager()
pending_jobs = []

# Gather jobs
for combination in product(model_names, feature_names, test_folds): # add any other loops

    # Unpack individual combinations
    model_name, features_col, test_fold = combination

    # Build position-based argument lists
    args = [
        data_path,
        desc_path,
        output_dir,
        model_name,
        smiles_col,
        features_col,
        target_col,
        fold_col,
        weights_col,
        groups_col,
        optim_metric,
        str(n_trials),
        str(n_jobs),
        str(test_fold),
        task
    ]

    # Job name
    name = '_'.join(map(str, combination))

    # Make logs dir
    log_dir = f"{output_dir}/{model_name}/{features_col}/logs"
    os.makedirs(log_dir, exist_ok=True)

    # Check if given job was already finished (for restarts)
    job_finished = False
    log_path = os.path.join(log_dir, f'job_{test_fold}.out')
    if os.path.exists(log_path):
        log_lines = [line.strip('\n') for line in open(log_path, 'r').readlines()]
        job_finished = 'Job finished' in log_lines

    # Executable command
    cmd = [script_path] + args

    job_dc = {
        'name': name,
        'exec': '/bin/bash',
        'args': cmd,
        'stdout': f"{log_dir}/job_{test_fold}.out",
        'stderr': f"{log_dir}/job_{test_fold}.err",
        'numCores': {"exact": n_jobs}
    }

    if not job_finished:
        pending_jobs.append(job_dc)

running_jobs = []
while pending_jobs or running_jobs:

    # if slots are available
    while pending_jobs and len(running_jobs) < max_concurrent_jobs:
        job_to_submit = pending_jobs.pop(0)
        jobs = Jobs()
        jobs.add(job_to_submit)
        manager.submit(jobs)
        running_jobs.append(job_to_submit['name'])
        log(f"Submitted job {job_to_submit['name']} ({len(running_jobs)}/{max_concurrent_jobs})")
        # Wait between submitting jobs in case something goes wrong
        time.sleep(1)

    log(f"{len(running_jobs)} jobs running, {len(pending_jobs)} jobs pending. Sleeping {sleep_interval}s...")
    time.sleep(sleep_interval)

    # check if some slots are now available
    still_running = []
    for job_name in running_jobs:
        status = manager.status(job_name)['jobs'][job_name]['data']['status']
        if status == "SUCCEED":
            log(f'Finished job: {job_name}')
        elif status == "EXECUTING":
            still_running.append(job_name)
        else:
            log(f"Issue encountered with job < {job_name} >\n{status}")
    running_jobs = still_running

manager.finish()
log('All jobs completed')
