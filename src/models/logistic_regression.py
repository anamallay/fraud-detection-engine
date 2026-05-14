"""
logistic_regression.py
======================
Logistic Regression - a linear baseline classifier.

It draws a straight decision boundary between Normal and Fraud. Fast to train
and highly interpretable, which makes it a good reference point: any model that
cannot beat Logistic Regression is not earning its complexity.
"""

from sklearn.linear_model import LogisticRegression

from config import RANDOM_SEED


def get_model() -> tuple:
    """Return (estimator, param_grid) for Logistic Regression.

    Grid: ``C`` is the inverse regularisation strength - smaller values mean
    stronger regularisation (a simpler model).
    """
    estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    param_grid = {"C": [0.1, 1.0, 10.0]}
    return estimator, param_grid