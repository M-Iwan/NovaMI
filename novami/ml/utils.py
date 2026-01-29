import importlib
import inspect
from typing import Dict

from novami.data.transform import DataTransformer, get_transformer_params
from novami.ml.params import *
from novami.ml.models import RegressorUnit, ClassifierUnit


def prepare_unit(model_name: str, model_params: Dict, features: str, task: str, n_jobs: int):
    """
    Infer correct parameters and prepare a unit based on model:features:task combination.

    Parameters
    ----------
    model_name: str
        A name of sklearn-syntax compatible model (i.e., the fit-predict)
    model_params: dict
        Hyperparameters to use for model
    features: str
        Name of features (e.g., CDDD, Morgan, etc.)
    task: str
        Training type: 'regression' or 'classification'
    n_jobs: int
        Number of CPUs to use during training

    Returns
    -------
    unit: Unit
        A Unit object with correct parameters
    """
    # Get model class
    model_class = get_model(model_name)

    # Add n_jobs, if possible
    model_params = add_n_jobs(model_params, model_class, n_jobs)

    try:
        model = model_class(**model_params)
    except Exception as e:
        raise ValueError(f'Incorrect parameters passed to < {model_name} > : {model_params}.\n{e}')

    transformer_params = get_transformer_params(features)
    transformer = DataTransformer(**transformer_params)

    if task == 'classification':
        unit = ClassifierUnit(model, transformer, use_proba=True)
    elif task == 'regression':
        unit = RegressorUnit(model, transformer)
    else:
        raise ValueError(f'Task should be either "classification" or "regression", got < {task} > instead.')

    return unit


def get_model(model_name: str):

    model_mapping = {
        'XGBClassifier': ('xgboost', 'XGBClassifier'),
        'XGBRegressor': ('xgboost', 'XGBRegressor'),

        'LGBMClassifier': ('lightgbm', 'LGBMClassifier'),
        'LGBMRegressor': ('lightgbm', 'LGBMRegressor'),

        'CatBoostClassifier': ('catboost', 'CatBoostClassifier'),
        'CatBoostRegressor': ('catboost', 'CatBoostRegressor'),

        'RandomForestClassifier': ('sklearn.ensemble', 'RandomForestClassifier'),
        'RandomForestRegressor': ('sklearn.ensemble', 'RandomForestRegressor'),
        'SVC': ('sklearn.svm', 'SVC'),
        'SVR': ('sklearn.svm', 'SVR'),
        'SGDRegressor': ('sklearn.linear_model', 'SGDRegressor'),
        'LogisticRegression': ('sklearn.linear_model', 'LogisticRegression'),
        'KNeighborsClassifier': ('sklearn.neighbors', 'KNeighborsClassifier'),
        'KNeighborsRegressor': ('sklearn.neighbors', 'KNeighborsRegressor'),
    }

    if model_name not in model_mapping.keys():
        raise ValueError(f'Unknown model: < {model_name} >. Please add it to get_model function '
                         f'in novami.ml.utils')

    module_name, class_name = model_mapping.get(model_name)
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(f"Could not import < {model_name} > from < {module_name} >\n> {e}")


def add_n_jobs(hparams: Dict, model_class: object, n_jobs: int = 1):
    """
    Add a parameter controlling the number of used CPUs to model hyperparameters.

    Parameters
    ----------
    hparams: Dict
        A dictionary with hyperparameters to use.
    model_class: object
        Model's class
    n_jobs: int
        Number of CPUs to use. Default is 1.

    Returns
    -------
    hparams: Dict
    """
    init_params = inspect.signature(model_class.__init__).parameters

    if 'thread_count' in init_params:
        hparams['thread_count'] = n_jobs

    elif 'n_jobs' in init_params:
        hparams['n_jobs'] = n_jobs

    else:
        pass

    return hparams
