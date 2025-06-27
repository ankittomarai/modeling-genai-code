import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV


def train_bin_classifier(df, feature_cols, label_col='Margin_Bin'):
    """
    Train a CatBoostClassifier to predict margin bins, using GridSearchCV for hyperparameter tuning.
    Returns the best trained model.
    """
    X = df[feature_cols]
    y = df[label_col]
    model = CatBoostClassifier(verbose=0, random_state=42)
    param_grid = {
        'iterations': [100, 200],
        'depth': [6, 8],
        'learning_rate': [0.05, 0.1]
    }
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    print(f"Best CatBoostClassifier params: {grid.best_params_}")
    return grid.best_estimator_


def predict_bin(model, df, feature_cols):
    """
    Predict margin bins for new data using the trained classifier.
    Returns predicted bin labels (as a numpy array).
    """
    X = df[feature_cols]
    return model.predict(X) 