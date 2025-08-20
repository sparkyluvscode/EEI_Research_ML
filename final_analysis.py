# ==============================================================================
# FINAL SCRIPT FOR: "Beyond Euler: An Explainable Machine Learning Framework..."
# DESCRIPTION: This script loads the final, clean 147-sample dataset,
#              engineers features, trains an XGBoost model, evaluates its
#              performance, and generates SHAP plots for explainability.
# DATA SOURCE: final_eei_data.csv
# ==============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import shap
import matplotlib.pyplot as plt

print("Libraries imported successfully.")

# --- 1. Load the Final Dataset ---
try:
    df = pd.read_csv('final_eei_data.csv')
    print("Dataset 'final_eei_data.csv' loaded successfully.")
    print(f"Total samples to be analyzed: {len(df)}")
except FileNotFoundError:
    print("\nERROR: 'final_eei_data.csv' not found.")
    print("Please make sure the data file is in the same folder as this script.")
    exit()

# --- 2. Feature Engineering ---
df['length_m'] = df['length_cm'] / 100.0
df['diameter_m'] = df['diameter_mm'] / 1000.0
df['G_feature'] = (df['diameter_m']**4) / (df['length_m']**2)
df = pd.get_dummies(df, columns=['pasta_type'], prefix='type')

# Define features and target.
features = [col for col in df.columns if col.startswith('type_') or col in ['length_m', 'diameter_m', 'G_feature']]
target = 'load_N'

X = df[features]
y = df[target]

print("\nFeature engineering complete.")
print("Features for model:", features)

# --- 3. Model Training with 5-Fold Cross-Validation ---
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores, rmse_scores = [], []

print(f"\nStarting model training with 5-fold cross-validation...")
for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    print(f"Fold {fold+1}: R² = {r2:.3f}, RMSE = {rmse:.3f} N")

# --- 4. Final Performance Evaluation ---
avg_r2 = np.mean(r2_scores)
avg_rmse = np.mean(rmse_scores)

print("\n-------------------------------------------")
print("Cross-Validation Training Complete.")
print(f"FINAL Average R² Score: {avg_r2:.3f}")
print(f"FINAL Average RMSE:     {avg_rmse:.3f} N")
print("-------------------------------------------")
print("Use these values to update your manuscript.")

# --- 5. Generate and Save Figures ---
print("\nGenerating and saving figures...")
model.fit(X, y) # Train model on all data for final plots

# --- Predicted vs. Actual Plot ---
y_full_pred = model.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(y, y_full_pred, alpha=0.7, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2)
plt.xlabel("Actual Critical Load (N)")
plt.ylabel("Predicted Critical Load (N)")
plt.title(f"Predicted vs. Actual Load (R² = {avg_r2:.2f})")
plt.grid(True)
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=300)
plt.close()
print("Saved 'predicted_vs_actual.png'")

# --- SHAP Summary Plot ---
explainer = shap.Explainer(model)
shap_values = explainer(X)
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title('SHAP Summary Plot: Global Feature Importance')
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300)
plt.close()
print("Saved 'shap_summary_plot.png'")

print("\nScript finished successfully.")
