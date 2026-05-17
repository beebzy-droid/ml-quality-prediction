# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:16:29 2026

@author: Bien
"""

# =============================================================
# Phase 4 Part 2 — Hyperparameter Tuning with Optuna
# =============================================================

import numpy as np
import pandas as pd
import optuna
import joblib
import warnings
from pathlib import Path

from sklearn.model_selection  import cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.multioutput      import MultiOutputRegressor
from sklearn.metrics          import r2_score, mean_squared_error
from xgboost                  import XGBRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
data_path = BASE / 'data'   / 'synthetic_process_data.csv'
model_dir = BASE / 'models'

# ── Load data + features ──────────────────────────────────────
df = pd.read_csv(data_path)

OUTPUTS = ['moisture', 'viscosity', 'purity', 'color_deviation']

df['temp_x_residence']    = df['temperature'] * df['residence_time']
df['temp_x_speed']        = df['temperature'] * df['mixing_speed']
df['speed_x_composition'] = df['mixing_speed'] * df['feed_composition']
df['temp_squared']        = df['temperature'] ** 2
df['residence_squared']   = df['residence_time'] ** 2

FEATURES = [
    'temperature', 'mixing_speed', 'residence_time', 'feed_composition',
    'temp_x_residence', 'temp_x_speed', 'speed_x_composition',
    'temp_squared', 'residence_squared'
]

X = df[FEATURES].values
y = df[OUTPUTS].values

from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42)

print(f"✅ Data loaded — Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")
print("\n── Tuning XGBoost with Optuna (50 trials) ──")
print("   This will take ~2 minutes...")

# ── Optuna objective ──────────────────────────────────────────
def objective(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 200, 600),
        'max_depth'       : trial.suggest_int('max_depth', 4, 10),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma'           : trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha'       : trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda'      : trial.suggest_float('reg_lambda', 0.5, 2.0),
        'random_state'    : 42,
        'verbosity'       : 0,
    }

    model = MultiOutputRegressor(XGBRegressor(**params))
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_vl_sc = scaler.transform(X_val)

    model.fit(X_tr_sc, y_train)
    y_pred = model.predict(X_vl_sc)

    # Optimize mean R² across all 4 targets
    r2 = r2_score(y_val, y_pred, multioutput='raw_values')
    return r2.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"\n   Best mean R² (val): {study.best_value:.4f}")
print(f"   Best params: {study.best_params}")

# ── Train final tuned XGBoost on train+val ────────────────────
print("\n── Training final tuned model ──")

best_params = study.best_params
best_params['random_state'] = 42
best_params['verbosity']    = 0

scaler_final = StandardScaler()
X_trainval   = np.vstack([X_train, X_val])
y_trainval   = np.vstack([y_train, y_val])

X_trainval_sc = scaler_final.fit_transform(X_trainval)
X_test_sc     = scaler_final.transform(X_test)

tuned_model = MultiOutputRegressor(XGBRegressor(**best_params))
tuned_model.fit(X_trainval_sc, y_trainval)

y_pred_test = tuned_model.predict(X_test_sc)

# ── Evaluate ──────────────────────────────────────────────────
test_r2   = r2_score(y_test, y_pred_test, multioutput='raw_values')
test_rmse = np.sqrt(mean_squared_error(
    y_test, y_pred_test, multioutput='raw_values'))

print("\n" + "="*55)
print("TUNED XGBOOST — Final Test Set Results")
print("="*55)
print(f"{'Target':<22} {'R²':>8} {'RMSE':>10}  {'KPI':>8}")
print("-"*55)

kpi_rmse = [0.5, 5.0, 0.3, 2.0]
for i, (out, kpi) in enumerate(zip(OUTPUTS, kpi_rmse)):
    status = '✅' if test_rmse[i] <= kpi else '❌'
    print(f"{out:<22} {test_r2[i]:>8.4f} {test_rmse[i]:>10.4f} "
          f" {status} ≤{kpi}")

print(f"\n{'Mean R²':<22} {test_r2.mean():>8.4f}")

# ── Save tuned model ──────────────────────────────────────────
joblib.dump({'model': tuned_model, 'scaler': scaler_final},
            model_dir / 'tuned_xgboost.pkl')
joblib.dump(FEATURES, model_dir / 'features.pkl')

print(f"\n✅ Tuned model saved → models/tuned_xgboost.pkl")
print("✅ Phase 4 tuning complete — ready for Phase 5 (SHAP)")