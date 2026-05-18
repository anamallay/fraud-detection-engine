# adaboost.py
from sklearn.ensemble import AdaBoostClassifier

from config import RANDOM_SEED


def get_model() -> tuple:

    estimator = AdaBoostClassifier(random_state=RANDOM_SEED)
    param_grid = {"n_estimators": [50, 100, 200]}
    return estimator, param_grid