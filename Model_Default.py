import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from config import dataset_dir, null_handler_option
from null_handler import null_handler
import numpy as np

"""Model_default is the file that runs our dataset through several independent scikit-learn models and compares their 
results."""

# Load and preprocess data
data = pd.read_csv(dataset_dir)

# Remove all null values from the data with null_killer
data = null_handler(null_handler_option, data)

# Verify the split
if data.shape[0] > 0:
    X = data.drop(columns=['Ag_P_balance_kg_ha'])
    y = data['Ag_P_balance_kg_ha']
else:
    raise ValueError("Dataset is empty after preprocessing. Please check the dataset or preprocessing steps.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data for Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Neural Network': MLPRegressor(random_state=42, max_iter=500),
    'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror'),
}

# Hyperparameters for tuning (example values)
param_grid = {
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],  # Regularization parameter
        'learning_rate_init': [0.001, 0.01],
    },
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
}

# Training and evaluation function
results = {}
feature_importance = {}

for name, model in models.items():
    print(f"Training {name}...")

    # Use scaled data only for Neural Network
    if name == 'Neural Network':
        X_train_input, X_test_input = X_train_scaled, X_test_scaled
    else:
        X_train_input, X_test_input = X_train, X_test

    # Use GridSearchCV if needed
    if name in param_grid:
        grid_search = GridSearchCV(model, param_grid[name], cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train_input, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train_input, y_train)

    # Predict and evaluate
    y_pred = best_model.predict(X_test_input)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2 Score': r2,
                     'Best Params': grid_search.best_params_ if name in param_grid else None}

    # Feature importance or coefficients
    if hasattr(best_model, 'feature_importances_'):
        feature_importance[name] = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        feature_importance[name] = np.abs(best_model.coef_)

# Display results and feature importance
for model_name, metrics in results.items():
    print(f"\n{model_name} - MSE: {metrics['MSE']}, R2 Score: {metrics['R2 Score']}")
    if metrics['Best Params']:
        print(f"Best Params: {metrics['Best Params']}")

    if model_name in feature_importance:
        print(f"Feature Importance (or Coefficients):")
        importance = feature_importance[model_name]
        sorted_idx = np.argsort(importance)[::-1]  # Sort features by importance
        for i in sorted_idx:
            print(f"Feature {X.columns[i]}: {importance[i]}")
