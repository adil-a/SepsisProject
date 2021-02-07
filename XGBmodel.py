from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
import fileio
from matplotlib import pyplot as plt
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from typing import Dict, Any
import os

ONE_DIR_UP = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
dftrainX = fileio.pklOpener(ONE_DIR_UP + '/trainSetDForiginal.pkl')
dftestX = fileio.pklOpener(ONE_DIR_UP + '/testSetDForiginal.pkl')
dftrainY = dftrainX['SepsisLabel']
dftestY = dftestX['SepsisLabel']
dftrainX.drop(['SepsisLabel', 'Filename'], axis=1, inplace=True)
dftestX.drop(['SepsisLabel', 'Filename'], axis=1, inplace=True)
columns = list(dftrainX.columns)
trainX = dftrainX.to_numpy()
trainY = dftrainY.to_numpy().reshape(dftrainY.shape[0], 1)
testX = dftestX.to_numpy()
testY = dftestY.to_numpy().reshape(dftestY.shape[0], 1)

def xgb_trainer(config: Dict[str, Any]) -> None:
    """
    Callable used for hyperparameter tuning

    Args:
        config: a dictionary containing generated hyperparameter values
    
    Returns:
        None
    """
    clf = xgboost.XGBClassifier(**config)
    clf.fit(trainX, trainY, eval_set=[(testX, testY)], eval_metric="auc", early_stopping_rounds=100, verbose=False)
    tune.report(auc=clf.evals_result()['validation_0']['auc'][-1], done=True)

def tuning() -> None:
    """
    Callable to start the hyperparameter tuning

    Args:
        None
    
    Returns:
        None
    """
    config = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": tune.randint(50, 1000),
        "reg_lambda": tune.uniform(0.01, 3),
        "max_depth": tune.randint(1, 10),
        "min_child_weight": tune.randint(1, 10),
        "subsample": tune.uniform(0.5, 1),
        "eta": tune.loguniform(1e-4, 1e-1),
        "gamma": tune.uniform(0, 10),
        "colsample_bytree": tune.uniform(0.5, 1)
    }

    optuna_search = OptunaSearch(
        metric="auc",
        mode="max"
    )

    analysis = tune.run(
        xgb_trainer,
        search_alg=optuna_search,
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=50,
        metric="auc",
        mode="max"
    )

tuning()