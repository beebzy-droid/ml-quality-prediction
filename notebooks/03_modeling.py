# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:15:02 2026

@author: Bien
"""

# =============================================================
# Phase 4 — Feature Engineering + Modeling
# Multi-output regression pipeline
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import joblib
from pathlib import Path

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing       import StandardScaler
from sklearn.pipeline            import Pipeline
from sklearn.multioutput         import MultiOutputRegressor
from sklearn.linear_model        import LinearRegression
from sklearn.ensemble            import RandomForestRegressor
from sklearn.neural_network      import MLPRegressor
from sklearn.metrics             import r2_score, mean_squared_error
from xgboost                     import XGBRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
data_path = BASE / 'data'  / 'synthetic_process_data.csv'
model_dir = BASE / 'models'

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv(data_path)

INPUTS  = ['temperature', 'mixing_speed',
           'residence_time', 'feed_composition']
OUTPUTS = ['moisture', 'viscosity', 'purity', 'color_deviation']

print(f"✅ Loaded: {df.shape}")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────
print("\n── Feature Engineering ──")

# Interaction terms from process physics
df['temp_x_residence']    = df['temperature'] * df['residence_time']
df['temp_x_speed']        = df['temperature'] * df['mixing_speed']
df['speed_x_composition'] = df['mixing_speed'] * df['feed_composition']
df['temp_squared']        = df['temperature'] ** 2
df['residence_squared']   = df['residence_time'] ** 2

FEATURES = INPUTS + [
    'temp_x_residence',
    'temp_x_speed',
    'speed_x_composition',
    'temp_squared',
    'residence_squared'
]

print(f"   Original features : {len(INPUTS)}")
print(f"   Engineered features: {len(FEATURES) - len(INPUTS)}")
print(f"   Total features    : {len(FEATURES)}")

X = df[FEATURES]
y = df[OUTPUTS]

# ── 3. TRAIN / VAL / TEST SPLIT ───────────────────────────────
print("\n── Train / Val / Test Split ──")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42)

print(f"   Train : {X_train.shape[0]} samples")
print(f"   Val   : {X_val.shape[0]} samples")
print(f"   Test  : {X_test.shape[0]} samples")

# ── 4. DEFINE MODELS ──────────────────────────────────────────
scaler = StandardScaler()

models = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  MultiOutputRegressor(LinearRegression()))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        ))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
        ))
    ]),
    'MLP Neural Network': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ))
    ]),
}

# ── 5. TRAIN AND EVALUATE ─────────────────────────────────────
print("\n── Training models ──")

results = {}

for name, pipeline in models.items():
    print(f"\n   Training {name}...")
    pipeline.fit(X_train, y_train)

    y_pred_val  = pipeline.predict(X_val)
    y_pred_test = pipeline.predict(X_test)

    val_r2   = r2_score(y_val,  y_pred_val,  multioutput='raw_values')
    test_r2  = r2_score(y_test, y_pred_test, multioutput='raw_values')
    test_rmse = np.sqrt(mean_squared_error(
        y_test, y_pred_test, multioutput='raw_values'))

    results[name] = {
        'pipeline'  : pipeline,
        'val_r2'    : val_r2,
        'test_r2'   : test_r2,
        'test_rmse' : test_rmse,
        'y_pred_test': y_pred_test
    }

    print(f"   {'Target':<20} {'R²':>8} {'RMSE':>10}")
    print(f"   {'-'*40}")
    for i, out in enumerate(OUTPUTS):
        print(f"   {out:<20} {test_r2[i]:>8.4f} {test_rmse[i]:>10.4f}")

# ── 6. MODEL COMPARISON TABLE ─────────────────────────────────
print("\n" + "="*60)
print("MODEL COMPARISON — Test Set R² per target")
print("="*60)

header = f"{'Model':<22}" + "".join(
    [f"{o[:8]:>10}" for o in OUTPUTS])
print(header)
print("-"*60)

for name, res in results.items():
    row = f"{name:<22}" + "".join(
        [f"{r2:>10.4f}" for r2 in res['test_r2']])
    print(row)

print("\nMODEL COMPARISON — Test Set RMSE per target")
print("="*60)
print(header)
print("-"*60)

for name, res in results.items():
    row = f"{name:<22}" + "".join(
        [f"{r:>10.4f}" for r in res['test_rmse']])
    print(row)

# ── 7. PREDICTED VS ACTUAL PLOTS ──────────────────────────────
print("\n── Predicted vs Actual plots ──")

best_model_name = max(results,
    key=lambda n: results[n]['test_r2'].mean())
print(f"   Best model: {best_model_name}")

best = results[best_model_name]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(f'Predicted vs Actual — {best_model_name} (Test Set)',
             fontsize=13)

for i, (ax, col) in enumerate(zip(axes, OUTPUTS)):
    actual = y_test.iloc[:, i].values
    pred   = best['y_pred_test'][:, i]
    r2     = best['test_r2'][i]
    rmse   = best['test_rmse'][i]

    ax.scatter(actual, pred, alpha=0.3, s=8,
               color='#5b8dd9', edgecolors='none')

    lims = [min(actual.min(), pred.min()),
            max(actual.max(), pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, alpha=0.8)

    ax.set_xlabel(f'Actual {col}', fontsize=9)
    ax.set_ylabel(f'Predicted {col}', fontsize=9)
    ax.set_title(f'R²={r2:.3f}  RMSE={rmse:.3f}', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(data_path.parent / 'model_01_pred_vs_actual.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("   Saved → model_01_pred_vs_actual.png")

# ── 8. SAVE BEST MODEL ────────────────────────────────────────
best_pipeline = results[best_model_name]['pipeline']
save_path = model_dir / 'best_model.pkl'
joblib.dump(best_pipeline, save_path)
print(f"\n✅ Best model saved → {save_path}")

# ── 9. SAVE FEATURE LIST ──────────────────────────────────────
feature_path = model_dir / 'features.pkl'
joblib.dump(FEATURES, feature_path)
print(f"✅ Feature list saved → {feature_path}")

print("\n✅ Phase 4 complete — ready for Phase 5 (Evaluation + SHAP)")