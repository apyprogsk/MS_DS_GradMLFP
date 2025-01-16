from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from config import dataset_dir, null_handler_option
from null_handler import null_handler
import numpy as np

"""Model_Stack is the file that runs our dataset through an ensemble of several scikit-learn models to combine their
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

# Define scalers for Neural Network, we've seen poor performance on the data as is, so were normalizing for the NN alone
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the base models
rf_model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=20)
xgb_model = XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1, objective='reg:squarederror')
nn_model = MLPRegressor(random_state=42, max_iter=500, hidden_layer_sizes=(100,), activation='relu')

# Train the base models separately
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
nn_model.fit(X_train_scaled, y_train)  # Use scaled data for Neural Network

# Evaluate the Neural Network
y_pred_nn = nn_model.predict(X_test_scaled)  # Use scaled data for testing
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print(f"\nNeural Network - MSE: {mse_nn}, R2 Score: {r2_nn}")

# Define the stacking ensemble model
stacked_model = StackingRegressor(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    final_estimator=LinearRegression(),
    cv=3
)

# Train the stacked model (uses original, unscaled data)
stacked_model.fit(X_train, y_train)

# Evaluate the stacked model
y_pred_stack = stacked_model.predict(X_test)
mse_stack = mean_squared_error(y_test, y_pred_stack)
r2_stack = r2_score(y_test, y_pred_stack)

print(f"\nStacked Model - MSE: {mse_stack}, R2 Score: {r2_stack}")

# Access feature importance from the base models (RandomForest and XGBoost)
rf_feature_importance = rf_model.feature_importances_
xgb_feature_importance = xgb_model.feature_importances_

# Combine feature importance from both base models
combined_feature_importance = rf_feature_importance + xgb_feature_importance

# Display feature importance from the base models
print("\nFeature Importance from Base Models:")
for i, feature in enumerate(X.columns):
    print(f"Feature {feature}: {combined_feature_importance[i]}")

# Feature importance or coefficients from the final estimator (LinearRegression)
feature_importance = {}

# Access the coefficients from the final estimator (LinearRegression)
if hasattr(stacked_model.final_estimator_, 'coef_'):
    feature_importance = np.abs(stacked_model.final_estimator_.coef_)

# Display feature importance from the final estimator
if feature_importance is not None:
    print("\nLinear Regression Coefficients (Final Estimator):")
    sorted_idx = np.argsort(feature_importance)[::-1]  # Sort features by importance
    for i in sorted_idx:
        print(f"Feature {X.columns[i]}: {feature_importance[i]}")
