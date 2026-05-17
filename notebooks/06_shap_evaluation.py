# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:22:58 2026

@author: Bien
"""

# =============================================================
# Phase 5 — SHAP Explainability + Evaluation
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics         import r2_score, mean_squared_error

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
data_path = BASE / 'data'   / 'synthetic_process_data.csv'
model_dir = BASE / 'models'
plot_dir  = BASE / 'data'

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv(data_path)
OUTPUTS = ['moisture', 'viscosity', 'purity', 'color_deviation']

# Rebuild features
df['temp_x_residence']    = df['temperature'] * df['residence_time']
df['temp_x_speed']        = df['temperature'] * df['mixing_speed']
df['speed_x_composition'] = df['mixing_speed'] * df['feed_composition']
df['temp_squared']        = df['temperature'] ** 2
df['residence_squared']   = df['residence_time'] ** 2
df['temp_centered']       = (df['temperature'] - 90) ** 2
df['residence_centered']  = (df['residence_time'] - 35) ** 2
df['temp_res_interaction'] = df['temperature'] * df['residence_time'] * 0.1
df['temp_x_composition']  = df['temperature'] * df['feed_composition']
df['temp_cubed_norm']     = (df['temperature'] / 100) ** 3
df['composition_squared'] = df['feed_composition'] ** 2
df['comp_x_temp']         = df['feed_composition'] * df['temperature']

FEATURES = [
    'temperature', 'mixing_speed', 'residence_time', 'feed_composition',
    'temp_x_residence', 'temp_x_speed', 'speed_x_composition',
    'temp_squared', 'residence_squared', 'temp_centered',
    'residence_centered', 'temp_res_interaction', 'temp_x_composition',
    'temp_cubed_norm', 'composition_squared', 'comp_x_temp',
]

X = df[FEATURES].values
y = df[OUTPUTS].values

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42)

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.vstack([y_train, y_val])

# ── Load final model ──────────────────────────────────────────
saved      = joblib.load(model_dir / 'final_model.pkl')
model      = saved['model']
scaler     = saved['scaler']

X_test_sc     = scaler.transform(X_test)
X_trainval_sc = scaler.transform(X_trainval)

y_pred = model.predict(X_test_sc)

test_r2   = r2_score(y_test, y_pred, multioutput='raw_values')
test_rmse = np.sqrt(mean_squared_error(
    y_test, y_pred, multioutput='raw_values'))

print("✅ Model loaded")
print("\n── Test Set Performance ──────────────────────────")
for i, out in enumerate(OUTPUTS):
    print(f"   {out:<22} R²={test_r2[i]:.4f}  RMSE={test_rmse[i]:.4f}")

# ── Plot 1: Predicted vs Actual (all 4 targets) ───────────────
print("\n── Plot 1: Residual Analysis ──")

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Residual Analysis — Final Model (Test Set)', fontsize=13)

for i, out in enumerate(OUTPUTS):
    actual    = y_test[:, i]
    predicted = y_pred[:, i]
    residuals = actual - predicted

    # Predicted vs Actual
    ax1 = axes[0][i]
    ax1.scatter(actual, predicted, alpha=0.3, s=8,
                color='#5b8dd9', edgecolors='none')
    lims = [min(actual.min(), predicted.min()),
            max(actual.max(), predicted.max())]
    ax1.plot(lims, lims, 'r--', linewidth=1.5)
    ax1.set_xlabel(f'Actual', fontsize=9)
    ax1.set_ylabel(f'Predicted', fontsize=9)
    ax1.set_title(f'{out.replace("_"," ").title()}\nR²={test_r2[i]:.3f}',
                  fontsize=10)
    ax1.spines[['top', 'right']].set_visible(False)

    # Residual distribution
    ax2 = axes[1][i]
    ax2.hist(residuals, bins=35, color='#e07b54',
             alpha=0.75, edgecolor='none')
    ax2.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax2.set_xlabel('Residual', fontsize=9)
    ax2.set_ylabel('Count', fontsize=9)
    ax2.set_title(f'RMSE={test_rmse[i]:.3f}', fontsize=10)
    ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(plot_dir / 'eval_01_residuals.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("   Saved → eval_01_residuals.png")

# ── Plot 2: SHAP Summary — one plot per target ────────────────
print("\n── Plot 2: SHAP Summary Plots ──")
print("   Computing SHAP values (takes ~60 seconds)...")

# Use 200 background samples for speed
background = shap.sample(
    pd.DataFrame(X_trainval_sc, columns=FEATURES), 200)

feature_labels = [f.replace('_', ' ') for f in FEATURES]

for i, out in enumerate(OUTPUTS):
    print(f"   Computing SHAP for {out}...")
    estimator = model.estimators_[i]
    explainer  = shap.TreeExplainer(estimator)
    shap_vals  = explainer.shap_values(
        pd.DataFrame(X_test_sc, columns=FEATURES))

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_vals,
        pd.DataFrame(X_test_sc, columns=FEATURES),
        feature_names=feature_labels,
        show=False,
        plot_size=None
    )
    plt.title(f'SHAP Feature Importance — {out.replace("_"," ").title()}',
              fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(plot_dir / f'eval_02_shap_{out}.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print(f"   Saved → eval_02_shap_{out}.png")

# ── Plot 3: SHAP Waterfall — single batch explanation ─────────
print("\n── Plot 3: SHAP Waterfall (single batch) ──")

sample_idx = 42  # pick one test batch to explain

for i, out in enumerate(OUTPUTS):
    estimator = model.estimators_[i]
    explainer  = shap.TreeExplainer(estimator)
    shap_vals  = explainer(
        pd.DataFrame(X_test_sc, columns=FEATURES))

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_vals[sample_idx], show=False)
    plt.title(f'Why did the model predict this {out}? — Batch #{sample_idx}',
              fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(plot_dir / f'eval_03_waterfall_{out}.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print(f"   Saved → eval_03_waterfall_{out}.png")

# ── What-if Analysis ──────────────────────────────────────────
print("\n── What-if Analysis ──────────────────────────────")
print("   Scenario: Reactor temperature increases by +10°C")
print("   (all other inputs held at dataset mean)\n")

# Baseline — all inputs at mean
means        = df[['temperature','mixing_speed',
                   'residence_time','feed_composition']].mean()
base_temp    = means['temperature']

def build_row(temp):
    row = {}
    row['temperature']         = temp
    row['mixing_speed']        = means['mixing_speed']
    row['residence_time']      = means['residence_time']
    row['feed_composition']    = means['feed_composition']
    row['temp_x_residence']    = temp * means['residence_time']
    row['temp_x_speed']        = temp * means['mixing_speed']
    row['speed_x_composition'] = means['mixing_speed'] * means['feed_composition']
    row['temp_squared']        = temp ** 2
    row['residence_squared']   = means['residence_time'] ** 2
    row['temp_centered']       = (temp - 90) ** 2
    row['residence_centered']  = (means['residence_time'] - 35) ** 2
    row['temp_res_interaction'] = temp * means['residence_time'] * 0.1
    row['temp_x_composition']  = temp * means['feed_composition']
    row['temp_cubed_norm']     = (temp / 100) ** 3
    row['composition_squared'] = means['feed_composition'] ** 2
    row['comp_x_temp']         = means['feed_composition'] * temp
    return pd.DataFrame([row])[FEATURES]

temps     = np.arange(60, 125, 5)
whatif    = {out: [] for out in OUTPUTS}

for t in temps:
    row_sc = scaler.transform(build_row(t))
    pred   = model.predict(row_sc)[0]
    for j, out in enumerate(OUTPUTS):
        whatif[out].append(pred[j])

# Print table
print(f"  {'Temp':>6} " +
      "".join([f"{o[:8]:>10}" for o in OUTPUTS]))
print("  " + "-"*50)
for k, t in enumerate(temps):
    marker = " ← baseline" if abs(t - base_temp) < 3 else ""
    print(f"  {t:>6.0f}°C " +
          "".join([f"{whatif[o][k]:>10.3f}" for o in OUTPUTS]) +
          marker)

# What-if plot
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('What-if Analysis — Effect of Reactor Temperature on Quality',
             fontsize=13)

units = ['%', 'cP', '%', 'ΔE']
for ax, out, unit in zip(axes, OUTPUTS, units):
    ax.plot(temps, whatif[out], color='#5b8dd9',
            linewidth=2.5, marker='o', markersize=4)
    ax.axvline(base_temp, color='#e07b54',
               linestyle='--', linewidth=1.5,
               label=f'Baseline {base_temp:.0f}°C')
    ax.set_xlabel('Reactor temperature (°C)', fontsize=9)
    ax.set_ylabel(f'{out.replace("_"," ").title()} ({unit})', fontsize=9)
    ax.set_title(out.replace('_', ' ').title(), fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(plot_dir / 'eval_04_whatif_temperature.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("\n   Saved → eval_04_whatif_temperature.png")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "="*55)
print("PHASE 5 COMPLETE — Evaluation Summary")
print("="*55)
print(f"\n{'Target':<22} {'R²':>8} {'RMSE':>10}")
print("-"*42)
for i, out in enumerate(OUTPUTS):
    print(f"{out:<22} {test_r2[i]:>8.4f} {test_rmse[i]:>10.4f}")
print(f"\n{'Mean R²':<22} {test_r2.mean():>8.4f}")
print("\n✅ All evaluation plots saved to data/")
print("✅ Ready for Phase 6 — Streamlit App")