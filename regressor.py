import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV


def train_regressor(df, feature_cols, target_col='Margin_Per_Unit'):
    """
    Train a CatBoostRegressor to predict margin per unit, using GridSearchCV for hyperparameter tuning.
    Returns the best trained model.
    """
    X = df[feature_cols]
    y = df[target_col]
    model = CatBoostRegressor(verbose=0, random_state=42)
    param_grid = {
        'iterations': [100, 200],
        'depth': [6, 8],
        'learning_rate': [0.05, 0.1]
    }
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X, y)
    print(f"Best CatBoostRegressor params: {grid.best_params_}")
    return grid.best_estimator_


def predict_margin(model, df, feature_cols):
    """
    Predict margin per unit for new data using the trained regressor.
    Returns predicted margins (as a numpy array).
    """
    X = df[feature_cols]
    return model.predict(X) 