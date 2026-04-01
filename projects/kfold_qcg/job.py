import argparse
import datetime
import os, joblib

import polars as pl
from novami.io import read_pl, write_pl
from novami.data.manager import KFoldManager
from novami.ml.optimize import optimize_unit_optuna
from novami.ml.params import *

parser = argparse.ArgumentParser(prog="OptunaOptimization")

parser.add_argument("--data_path", help="Path to data file.")
parser.add_argument("--desc_path", help="Path to descriptors file.")
parser.add_argument("--output_dir", help="Path to directory to save results")
parser.add_argument("--model_name", help="Name of the ML model to use")

parser.add_argument("--smiles_col", help="Name of the column with SMILES strings")
parser.add_argument("--features_col", help='Name of the column with features.')
parser.add_argument("--target_col", help='Name of the column with target.')
parser.add_argument("--fold_col", help='Name of the column with fold.', default='Fold')
parser.add_argument("--weights_col", help='Name of the column with sample weights.', default=None)
parser.add_argument("--groups_col", help='Name of the column with groups.', default=None)

parser.add_argument("--optim_metric", help="Metric used to guide Optuna's optimization")
parser.add_argument("--n_trials", help="Number of Optuna trials to run", type=int)
parser.add_argument("--n_jobs", help="Number of CPUs to use", type=int)
parser.add_argument("--test_fold", help="Fold to be used as a test set", type=int)
parser.add_argument("--task", help="Kind of model to run. Either regression or classification")

def main(args):
    def log(message: str):
        log_time = str(datetime.datetime.now()).split('.')[0]
        print(f'{log_time}: {message}')

    log('Job started')
    log('Checking paths')

    data_path = args.data_path
    desc_path = args.desc_path
    output_dir = args.output_dir
    model_name = args.model_name
    features_col = args.features_col
    test_fold = args.test_fold
    task = args.task

    if not os.path.exists(data_path):
        raise OSError(f"Data file at < {data_path} > does not exist")
    if not os.path.exists(desc_path):
        raise OSError(f"Descriptors file at < {desc_path} > does not exist")

    save_dir = str(os.path.join(output_dir, model_name, features_col))
    os.makedirs(save_dir, exist_ok=True)

    study_path = os.path.join(save_dir, f'study_{test_fold}.joblib')
    ensemble_path = os.path.join(save_dir, f'ensemble_{test_fold}.joblib')
    pred_path = os.path.join(save_dir, f'preds_{test_fold}.tsv')
    scores_path = os.path.join(save_dir, f'scores_{test_fold}.tsv')

    log('> Paths OK')
    log('Checking data')

    smiles_col = args.smiles_col
    target_col = args.target_col
    fold_col = args.fold_col
    weights_col = args.weights_col
    groups_col = args.groups_col

    required_cols = [smiles_col, features_col, target_col]

    # If weights or groups are not used, an empty string should be passed
    if weights_col == '':
        weights_col = None
    else:
        required_cols.append(weights_col)
    if groups_col == '':
        groups_col = None
    else:
        required_cols.append(groups_col)

    data = read_pl(data_path)
    desc = read_pl(desc_path)

    df = data.join(desc[[smiles_col, features_col]], on=smiles_col, how='inner')

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column < {col} > not found in dataframe")

    if test_fold not in df[fold_col].unique():
        raise ValueError(f"Test fold < {test_fold} > not found in dataframe")

    log('> Data OK')
    log('Checking parameters')

    optim_metric = args.optim_metric
    n_trials = args.n_trials
    n_jobs = args.n_jobs

    prm_fn = get_params_function(model_name)
    if prm_fn is None:
        raise ValueError(f"Model < {model_name} > not supported. Please add a corresponding "
                         f"parameter function to novami.ml.params")

    if task == 'classification':
        if optim_metric not in ["TP", "FP", "FN", "TN", "Accuracy", "Recall", "Specificity", "Precision",
                                "Balanced Accuracy", "GeomRS", "HarmRS", "F1 Score", "ROC AUC", "MCC",
                                'PRC AUC', 'ECE', 'Brier']:
            raise ValueError(f"Metric < {optim_metric} > not supported")
    elif task == 'regression':
        if optim_metric not in ['R2', 'RMSE', 'MAE']:
            raise ValueError(f"Metric < {optim_metric} > not supported")

    if not isinstance(n_trials, int) or n_trials < 1:
        raise TypeError("Number of trials must be a positive integer")

    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise TypeError("Number of jobs must be a positive integer")

    log('> Parameters OK')
    log('Preparing DatasetManager')

    manager = KFoldManager(
        df=df,
        smiles_col=smiles_col,
        features_col=features_col,
        target_col=target_col,
        fold_col=fold_col,
        test_fold=test_fold,
        weights_col=weights_col,
        groups_col=groups_col,
    )

    log('> KFoldManager OK')
    log('Performing Optuna optimization')

    study, ensemble = optimize_unit_optuna(
        model_name=model_name,
        dataset_manager=manager,
        optimization_metric=optim_metric,
        task=task,
        n_trials=n_trials,
        n_jobs=n_jobs
    )

    log('> Optimization OK')
    log('Saving results')

    joblib.dump(study, study_path)
    log(f'> Study saved at {study_path}')
    joblib.dump(ensemble, ensemble_path)
    log(f'> Ensemble saved at {ensemble_path}')

    log('Evaluating ensemble')

    test_data, test_smiles = manager.get_test_data(), manager.get_test_smiles()
    y_true = test_data['y_true']
    y_pred = ensemble.predict(test_data['x_array'])

    pred = pl.DataFrame({smiles_col: test_smiles, 'y_true': y_true, 'y_pred': y_pred})
    scores = ensemble.score(**test_data)

    log('> Evaluation OK')
    log('Saving results')

    write_pl(pred, pred_path)
    log(f'> Predictions saved at {pred_path}')
    write_pl(scores, scores_path)
    log(f'> Scores saved at {scores_path}')

    log('Job finished successfully')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)