# decision_tree.py

from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_SEED


def get_model() -> tuple:

    estimator = DecisionTreeClassifier(random_state=RANDOM_SEED)
    param_grid = {"max_depth": [5, 10, 20, None]}
    return estimator, param_grid