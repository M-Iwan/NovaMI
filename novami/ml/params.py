from typing import Dict
import optuna


def get_params_function(model_name: str):
    functions = {
        'XGBClassifier': get_params_xgb_classifier,
        'XGBRegressor': get_params_xgb_regressor,
        'LGBMRegressor': get_params_lgbm_regressor,
        'CatBoostRegressor': get_params_catboost_regressor,
        'RandomForestClassifier': get_params_random_forest_classifier,
        'RandomForestRegressor': get_params_random_forest_classifier,
        'SVC': get_params_svc,
        'SVR': get_params_svr,
        'SGDRegressor': get_params_sgd_regressor,
        'LogisticRegression': get_params_logistic_regression,
        'KNeighborsClassifier': get_params_knn_classifier,
        'KNeighborsRegressor': get_params_knn_regressor,
    }

    model_fn = functions.get(model_name, None)
    if model_fn is None:
        raise KeyError(f"No parameters function defined for < {model_name} >. "
                       f"Please update the < get_params_function > in novami.ml.params")

    return model_fn


def get_params_xgb_classifier(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for XGBClassifier for optuna
    """
    return {
        # high-impact parameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 1e-1, log=True),

        # medium-impact parameters
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),

        # low-impact parameters
        #'max_leaves': trial.suggest_int('max_leaves', 0, 12),
        #'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        #'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
    }


def get_params_xgb_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for XGBRegressor for optuna
    """
    return {
        # high-impact parameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 1e-1, log=True),

        # medium-impact parameters
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),

        # low-impact parameters
        #'max_leaves': trial.suggest_int('max_leaves', 0, 12),
        #'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        #'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
    }


def get_params_lgbm_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for LGBMRegressor for optuna, including boosting types
    """

    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])

    params = {
        # high-impact parameters
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 1e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'boosting_type': boosting_type,

        # medium-impact parameters
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),

        # low-impact parameters
        #'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        #'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        #'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        #'min_split_gain': trial.suggest_float('min_split_gain', 0, 5),
    }

    if boosting_type == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.1, 0.5)

    if boosting_type == 'goss':
        params.pop('subsample', None)  # Not used by GOSS
        params['top_rate'] = trial.suggest_float('top_rate', 0.1, 0.3)
        params['other_rate'] = trial.suggest_float('other_rate', 0.05, 0.2)

    return params

def get_params_catboost_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for CatBoostRegressor for optuna
    """
    return {
        # high-impact parameters
        'iterations': trial.suggest_int('iterations', 100, 1000, step=25),
        'depth': trial.suggest_int('depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 1e-1, log=True),

        # medium-impact parameters
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),

        # low-impact parameters
        #'random_strength': trial.suggest_float('random_strength', 0, 5),
        #'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        #'border_count': trial.suggest_int('border_count', 32, 255),
        #'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
    }


def get_params_random_forest_classifier(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for RandomForestClassifier for optuna
    """
    return {
        # high-impact parameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),

        # medium-impact parameters
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),

        # low-impact parameters
        #'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 0.05, log=True)
    }


def get_params_random_forest_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for RandomForestRegressor for optuna
    """
    return {
        # high-impact parameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),

        # medium-impact parameters
        'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error', 'poisson']),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),

        # low-impact parameters
        # 'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 0.05, log=True)
    }


def get_params_svc(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for SVC for optuna
    """
    # high-impact parameters
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    params = {
        'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
        'kernel': kernel
    }

    # medium-impact parameters
    if kernel in ['poly', 'rbf', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-2, 1e0, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)

    # low-impact parameters
    if kernel in ['poly', 'sigmoid']:
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)

    return params


def get_params_svr(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for SVR for optuna
    """
    # high-impact parameters
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    params = {
        'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
        'kernel': kernel,
        'epsilon': trial.suggest_float('epsilon', 1e-2, 1e0, log=True)
    }

    # medium-impact parameters
    if kernel in ['poly', 'rbf', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-3, 1.0, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)

    # low-impact parameters
    if kernel in ['poly', 'sigmoid']:
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)

    return params


def get_params_sgd_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for StochasticGradientDescentRegressor for optuna.
    Equivalent of testing LinearRegression, Lasso, Ridge, and ElasticNet
    """
    loss = trial.suggest_categorical('loss', [
        'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
    ])

    penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet', None])

    params = {
        # high-impact parameters
        'loss': loss,
        'penalty': penalty,
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),

        # medium-impact parameters
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),

        # low-impact parameters
        'max_iter': trial.suggest_int('max_iter', 500, 2000),
        'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True),
    }

    if penalty == 'elasticnet': # medium
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

    if loss in ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']: # low
        params['epsilon'] = trial.suggest_float('epsilon', 0.01, 0.5)

    if params['learning_rate'] in ['constant', 'invscaling', 'adaptive']: # low
        params['eta0'] = trial.suggest_float('eta0', 0.001, 0.1, log=True)

    if params['learning_rate'] == 'invscaling': # low
        params['power_t'] = trial.suggest_float('power_t', 0.1, 0.5)

    return params


def get_params_logistic_regression(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for LogisticRegression for optuna
    """
    valid_combinations = [
        'lbfgs:l2', 'lbfgs:None',
        'liblinear:l1', 'liblinear:l2',
        'newton-cg:l2', 'newton-cg:None',
        'newton-cholesky:l2', 'newton-cholesky:None',
        'sag:l2', 'sag:None', 'saga:elasticnet',
        'saga:l1', 'saga:l2', 'saga:None'
    ]
    combination = trial.suggest_categorical('solver_penalty', valid_combinations)
    solver, penalty = combination.split(':')
    penalty = None if penalty == 'None' else penalty

    params = {
        # high-impact parameters
        'solver': solver,
        'penalty': penalty,
        'C': trial.suggest_float('C', 0.001, 10, log=True),

        # medium-impact parameters
        'max_iter': 1024,
    }

    if penalty == 'elasticnet': # medium
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

    return params


def get_params_knn_classifier(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for KNeighborsClassifier for optuna
    """
    params = {
        # high-impact parameters
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 25),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),

        # medium-impact parameters
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 100),

        # low-impact parameters
        'p': trial.suggest_int('p', 1, 2)
    }

    return params

def get_params_knn_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for KNeighborsRegressor for optuna
    """
    params = {
        # high-impact parameters
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 25),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),

        # medium-impact parameters
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 100),

        # low-impact parameters
        'p': trial.suggest_int('p', 1, 2)
    }

    return params
