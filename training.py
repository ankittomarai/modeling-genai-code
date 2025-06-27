import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from catboost import CatBoostRegressor

def monthwise_validation(df, feature_cols, target_col='Margin_Per_Unit', months_back=6):
    """
    Perform month-wise validation for the last `months_back` months using CatBoost only.
    Returns validation metrics and the best model.
    """
    model = CatBoostRegressor(verbose=0, random_state=42)
    param_grid = {'iterations': [100, 200], 'depth': [6, 8], 'learning_rate': [0.05, 0.1]}
    last_month = df['YearMonth'].max()
    months = [last_month - i for i in range(months_back, 0, -1)]
    results = {'CatBoost': {'r2': [], 'mape': [], 'best_params': []}}
    best_model = None
    for month in months:
        train = df[df['YearMonth'] < month]
        test = df[df['YearMonth'] == month]
        if len(test) == 0 or len(train) == 0:
            continue
        X_train = train[feature_cols]
        y_train = train[target_col]
        X_test = test[feature_cols]
        y_test = test[target_col]
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        results['CatBoost']['r2'].append(r2)
        results['CatBoost']['mape'].append(mape)
        results['CatBoost']['best_params'].append(grid.best_params_)
        print(f"CatBoost | Month: {month} | R2: {r2:.4f} | MAPE: {mape:.4f} | Best Params: {grid.best_params_}")
    avg_r2 = np.mean(results['CatBoost']['r2'])
    avg_mape = np.mean(results['CatBoost']['mape'])
    print("\nAverage Validation Scores (last 6 months):")
    print(f"CatBoost: R2={avg_r2:.4f}, MAPE={avg_mape:.4f}")
    print(f"\nBest algorithm: CatBoost")
    return 'CatBoost', best_model, {'CatBoost': (avg_r2, avg_mape)}, results

def train_final_model(df, feature_cols, best_alg, param_grids, target_col='Margin_Per_Unit'):
    """
    Retrain CatBoost on all data except the latest month.
    Returns the trained model and the test set for the latest month.
    """
    model = CatBoostRegressor(verbose=0, random_state=42)
    latest_month = df['YearMonth'].max()
    train = df[df['YearMonth'] < latest_month]
    test = df[df['YearMonth'] == latest_month]
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    grid = GridSearchCV(model, param_grids['CatBoost'], cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"\nBest parameters for final model: {grid.best_params_}")
    return best_model, test 