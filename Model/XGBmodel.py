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
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# dftrainX_batch1 = fileio.pklOpener(THIS_FOLDER + '/trainingSetAugmentedDF_batch1.pkl')
# dftrainX_batch2 = fileio.pklOpener(THIS_FOLDER + '/trainingSetAugmentedDF_batch2.pkl')
dftrainX = fileio.pklOpener(THIS_FOLDER + '/trainingSetAugmentedDF.pkl')
dftestX = fileio.pklOpener(THIS_FOLDER + '/testSetAugmentedDF.pkl')
# dftrainY_batch1 = dftrainX_batch1['SepsisLabel']
# dftrainY_batch2 = dftrainX_batch2['SepsisLabel']
dftrainY = dftrainX['SepsisLabel']
dftestY = dftestX['SepsisLabel']
# dftrainX_batch1.drop(['SepsisLabel'], axis=1, inplace=True)
# dftrainX_batch2.drop(['SepsisLabel'], axis=1, inplace=True)
dftrainX.drop(['SepsisLabel'], axis=1, inplace=True)
dftestX.drop(['SepsisLabel'], axis=1, inplace=True)
columns = list(dftestX.columns)
# trainX_batch1 = dftrainX_batch1.to_numpy()
# trainX_batch2 = dftrainX_batch2.to_numpy()
# trainY_batch1 = dftrainY_batch1.to_numpy().reshape(dftrainY_batch1.shape[0], 1)
# trainY_batch2 = dftrainY_batch2.to_numpy().reshape(dftrainY_batch2.shape[0], 1)
trainX = dftrainX.to_numpy()
testX = dftestX.to_numpy()
trainY = dftrainY.to_numpy().reshape(dftrainY.shape[0], 1)
testY = dftestY.to_numpy().reshape(dftestY.shape[0], 1)

# lst = [(trainX_batch1, trainY_batch1), (trainX_batch2, trainY_batch2)]

# analysis_result = []
# for item in lst:
def xgb_trainer(config: Dict[str, Any]) -> None:
    """
    Callable used for hyperparameter tuning

    Args:
        config: a dictionary containing generated hyperparameter values
    
    Returns:
        None
    """
    clf = xgboost.XGBClassifier(**config)
    clf.fit(item[0], item[1], eval_set=[(testX, testY)], eval_metric="auc", early_stopping_rounds=100, verbose=False)
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
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=50,
        metric="auc",
        mode="max"
    )
    analysis_result.append((analysis.best_result, analysis.best_config))
# tuning()
lst = [{
    'objective': 'binary:logistic', 
    'eval_metric': 'auc', 
    'n_estimators': 979, 
    'reg_lambda': 0.5639286016008462, 
    'max_depth': 7, 
    'min_child_weight': 8, 
    'subsample': 0.6231423879265132, 
    'eta': 0.01103975961113523, 
    'gamma': 6.1353767319534755, 
    'colsample_bytree': 0.8657164209860049, 
    'n_jobs': -1}, 
    {
    'objective': 'binary:logistic', 
    'eval_metric': 'auc', 
    'n_estimators': 176, 
    'reg_lambda': 1.8663531146772794, 
    'max_depth': 8, 
    'min_child_weight': 3, 
    'subsample': 0.7558019549699201, 
    'eta': 0.036138077654744326, 
    'gamma': 9.410221487087785, 
    'colsample_bytree': 0.8802525498487996, 
    'n_jobs': -1}]
auc_scores = []
models = []
for config in lst:
    clf = xgboost.XGBClassifier(**config)
    clf.fit(trainX, trainY, eval_set=[(testX, testY)], eval_metric="auc", early_stopping_rounds=100, verbose=1)
    out_score = clf.predict_proba(testX)
    score = roc_auc_score(testY, out_score[:, 1])
    auc_scores.append(score)
    models.append(clf)

if auc_scores[0] > auc_scores[1]:
    fileio.StraightDumpDir(models[0], THIS_FOLDER + '/XGBmodel.pkl')
else:
    fileio.StraightDumpDir(models[1], THIS_FOLDER + '/XGBmodel.pkl')

print(auc_scores)
