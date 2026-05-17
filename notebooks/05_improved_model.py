# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:20:38 2026

@author: Bien
"""

# =============================================================
# Phase 4 Part 3 — Improved Features for Moisture + Purity
# =============================================================

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.multioutput     import MultiOutputRegressor
from sklearn.metrics         import r2_score, mean_squared_error
from xgboost                 import XGBRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE      = Path(__file__).parent.parent
data_path = BASE / 'data'   / 'synthetic_process_data.csv'
model_dir = BASE / 'models'

df = pd.read_csv(data_path)
OUTPUTS = ['moisture', 'viscosity', 'purity', 'color_deviation']

# ── Expanded feature set ──────────────────────────────────────
# Original
df['temp_x_residence']    = df['temperature'] * df['residence_time']
df['temp_x_speed']        = df['temperature'] * df['mixing_speed']
df['speed_x_composition'] = df['mixing_speed'] * df['feed_composition']
df['temp_squared']        = df['temperature'] ** 2
df['residence_squared']   = df['residence_time'] ** 2

# New — target purity's quadratic sweet spot directly
df['temp_centered']       = (df['temperature'] - 90) ** 2
df['residence_centered']  = (df['residence_time'] - 35) ** 2
df['temp_res_interaction'] = df['temperature'] * df['residence_time'] * 0.1

# New — target moisture more precisely
df['temp_x_composition']  = df['temperature'] * df['feed_composition']
df['temp_cubed_norm']     = (df['temperature'] / 100) ** 3

# New — target color deviation
df['composition_squared'] = df['feed_composition'] ** 2
df['comp_x_temp']         = df['feed_composition'] * df['temperature']

FEATURES = [
    # Original inputs
    'temperature', 'mixing_speed', 'residence_time', 'feed_composition',
    # Original interactions
    'temp_x_residence', 'temp_x_speed', 'speed_x_composition',
    'temp_squared', 'residence_squared',
    # Purity-targeted
    'temp_centered', 'residence_centered', 'temp_res_interaction',
    # Moisture-targeted
    'temp_x_composition', 'temp_cubed_norm',
    # Color-targeted
    'composition_squared', 'comp_x_temp',
]

print(f"✅ Total features: {len(FEATURES)}")

X = df[FEATURES].values
y = df[OUTPUTS].values

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42)

# ── Load best Optuna params from previous run ─────────────────
prev = joblib.load(model_dir / 'tuned_xgboost.pkl')

# Re-extract params from one of the estimators
sample_estimator = prev['model'].estimators_[0]
params = sample_estimator.get_params()

# Clean params for reuse
keep = ['n_estimators','max_depth','learning_rate','subsample',
        'colsample_bytree','min_child_weight','gamma',
        'reg_alpha','reg_lambda']
best_params = {k: params[k] for k in keep if k in params}
best_params['random_state'] = 42
best_params['verbosity']    = 0

print(f"   Reusing tuned params from previous run")

# ── Train on full train+val set ───────────────────────────────
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.vstack([y_train, y_val])

scaler = StandardScaler()
X_trainval_sc = scaler.fit_transform(X_trainval)
X_test_sc     = scaler.transform(X_test)

model = MultiOutputRegressor(XGBRegressor(**best_params))
model.fit(X_trainval_sc, y_trainval)

y_pred = model.predict(X_test_sc)

# ── Results ───────────────────────────────────────────────────
test_r2   = r2_score(y_test, y_pred, multioutput='raw_values')
test_rmse = np.sqrt(mean_squared_error(
    y_test, y_pred, multioutput='raw_values'))

# Revised KPIs
kpi_rmse   = [0.5, 13.6, 0.3, 2.0]
kpi_labels = ['≤0.5%', '≤3% rel', '≤0.3%', '≤2 ΔE']

print("\n" + "="*60)
print("IMPROVED MODEL — Final Test Set Results")
print("="*60)
print(f"{'Target':<22} {'R²':>8} {'RMSE':>10}  {'KPI'}")
print("-"*60)

for i, (out, kpi, label) in enumerate(
        zip(OUTPUTS, kpi_rmse, kpi_labels)):
    status = '✅' if test_rmse[i] <= kpi else '❌'
    print(f"{out:<22} {test_r2[i]:>8.4f} "
          f"{test_rmse[i]:>10.4f}  {status} {label}")

print(f"\n{'Mean R²':<22} {test_r2.mean():>8.4f}")

# ── Save final model ──────────────────────────────────────────
joblib.dump(
    {'model': model, 'scaler': scaler, 'features': FEATURES},
    model_dir / 'final_model.pkl'
)
joblib.dump(FEATURES, model_dir / 'features.pkl')

print(f"\n✅ Final model saved → models/final_model.pkl")
print("✅ Ready for Phase 5 — SHAP + Evaluation")