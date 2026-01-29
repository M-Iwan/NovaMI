import os
import argparse
from itertools import product

import numpy as np
import polars as pl
from novami.train.optimize import *
from novami.io.file import read_pl, write_pl

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


parser = argparse.ArgumentParser(prog='Optuna_HP', description='Perform HP-tuning using Optuna using a nested CV setup')

parser.add_argument('--data_file', help='Path to a .tsv file with SMILES and target values.')
parser.add_argument('--features_file', help='Path to a .joblib file with SMILES:features mapping.')
parser.add_argument('--smiles_col', help='Name of the SMILES columns')
parser.add_argument('--target_col', help='Name of the column with target values')
parser.add_argument('--weights_col', help='Name of the column with sample weights')
parser.add_argument('--fold_col', help='Name of the column with fold assignments')
parser.add_argument('--n_trials', help='Number of Optuna optimization steps', type=int)
parser.add_argument('--metric', help='Name of the metric guiding optimization')
parser.add_argument('--task', help='Either regression or classification')
parser.add_argument('--n_cpus', help='Number of CPUs to use', type=int)
parser.add_argument('--output_dir', help='Path to the output directory')

parsed_args = parser.parse_args()


def main(args):

    data_file = args.data_file
    features_file = args.features_file
    smiles_col = args.smiles_col
    target_col = args.target_col
    weights_col = args.weights_col
    fold_col = args.fold_col
    n_trials = args.n_trials
    metric = args.metric
    task = args.task
    n_cpus = args.n_cpus
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_file):
        raise ValueError(f'Data file at {data_file} does not exist')

    if not os.path.exists(features_file):
        raise ValueError(f'Descriptors file at {features_file} does not exist')

    data_df = read_pl(data_file)
    descriptors_df = read_pl(features_file)

    if smiles_col not in data_df.columns:
        raise ValueError(f'SMILES column {smiles_col} not found in data file')

    if smiles_col not in descriptors_df.columns:
        raise ValueError(f'SMILES column {smiles_col} not found in descriptors file')

    if target_col not in data_df.columns:
        raise ValueError(f'Target column {target_col} not found in data file')

    if weights_col == '':
        weights_col = None
    else:
        if weights_col not in data_df.columns:
            raise ValueError(f'Weights column {weights_col} not found in data file')

    if fold_col not in data_df.columns:
        raise ValueError(f'Fold column {fold_col} not found in data file')

    if n_trials <= 0:
        raise ValueError(f'Number of Optuna optimization steps must be greater than 0')

    if task not in ['regression', 'classification']:
        raise ValueError(f'Task {task} not supported')

    if metric not in ['Accuracy', 'Recall', 'Specificity', 'Precision', 'Balanced Accuracy',
        'GeomRS', 'HarmRS', 'F1 Score', 'ROC AUC', 'MCC', 'R2', 'MAE', 'RMSE']:
        raise ValueError(f'Metric {metric} not recognized')

    models = [LGBMRegressor, CatBoostRegressor, XGBRegressor, SGDRegressor,
              RandomForestRegressor, KNeighborsRegressor, SVR]

    descriptors = ['RDKit', 'Morgan', 'MACCS', 'Klek', 'ChemBERTa', 'CDDD']

    for model_class, desc in product(models, descriptors):
        model_name = model_class.__name__
        print(f'> Now processing: {model_name} and {desc}')
        try:
            df = data_df.join(descriptors_df[[smiles_col, desc]], on=smiles_col, how='inner')
            try:
                for fold in df[fold_col].unique():
                    print(f' > Processing fold: {fold}')
                    train_df = df.filter(pl.col(fold_col) != fold)
                    test_df = df.filter(pl.col(fold_col) == fold)

                    train_mgr = DatasetManager(
                        df=train_df,
                        features_col=desc,
                        target_col=target_col,
                        weights_col=weights_col,
                        fold_col=fold_col,
                    )

                    test_mgr = DatasetManager(
                        df=test_df,
                        features_col=desc,
                        target_col=target_col,
                        weights_col=weights_col,
                        fold_col=fold_col,
                    )

                    try:
                        save_dir = os.path.join(output_dir, model_name, desc, str(fold))
                        os.makedirs(save_dir, exist_ok=True)
                        study, ensemble = optimize_unit_optuna(
                            model_class=model_class,
                            dataset_manager=train_mgr,
                            selection_metric=metric,
                            task=task,
                            n_trials=n_trials,
                            n_cpus=n_cpus
                        )

                        train_scores = ensemble.metrics

                        test_data = test_mgr.get_eval_data(fold)
                        test_scores = ensemble.score(**test_data)

                        write_pl(study, f'{save_dir}/study.joblib')
                        write_pl(ensemble, f'{save_dir}/ensemble.joblib')
                        write_pl(train_scores, f'{save_dir}/train_scores.tsv')
                        write_pl(test_scores, f'{save_dir}/test_scores.tsv')

                    except Exception as e:
                        print(f'Exception during model training for {model_name} and {desc}:\n{e}')
            except Exception as e:
                print(f'Exception while preparing dataset manager for {desc}:\n{e}')
        except Exception as e:
            print(f'Exception while concatenating data and features for {desc}:\n{e}')

if __name__ == '__main__':
    main(parsed_args)
