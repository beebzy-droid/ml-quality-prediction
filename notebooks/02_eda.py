# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:12:23 2026

@author: Bien
"""

# =============================================================
# Phase 3 — Exploratory Data Analysis (EDA)
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path

# ── Load data ─────────────────────────────────────────────────
data_path = Path(__file__).parent.parent / 'data' / 'synthetic_process_data.csv'
df = pd.read_csv(data_path)

output_dir = Path(__file__).parent.parent / 'data'

INPUTS  = ['temperature', 'mixing_speed', 'residence_time', 'feed_composition']
OUTPUTS = ['moisture', 'viscosity', 'purity', 'color_deviation']

print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3))

# ── Plot 1: Correlation Heatmap ───────────────────────────────
print("\n── Plot 1: Correlation Heatmap ──")

corr = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    ax=ax,
    annot_kws={'size': 10}
)
ax.set_title('Correlation Matrix — Process Inputs vs Quality Outputs',
             fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(output_dir / 'eda_01_correlation_heatmap.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("   Saved → eda_01_correlation_heatmap.png")

# ── Plot 2: Input vs Output Scatter Plots ─────────────────────
print("\n── Plot 2: Input vs Output Scatter Matrix ──")

fig, axes = plt.subplots(len(OUTPUTS), len(INPUTS),
                          figsize=(18, 14))
fig.suptitle('Process Inputs vs Quality Outputs — Scatter Matrix',
             fontsize=14, y=1.01)

for i, out in enumerate(OUTPUTS):
    for j, inp in enumerate(INPUTS):
        ax = axes[i][j]
        ax.scatter(df[inp], df[out], alpha=0.15, s=6,
                   color='#5b8dd9', edgecolors='none')

        # Fit and plot trend line
        z = np.polyfit(df[inp], df[out], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[inp].min(), df[inp].max(), 100)
        ax.plot(x_line, p(x_line), color='#e07b54',
                linewidth=1.5, alpha=0.9)

        # Correlation value
        r, _ = stats.pearsonr(df[inp], df[out])
        ax.set_title(f'r = {r:.2f}', fontsize=9, color='#333')

        if i == len(OUTPUTS) - 1:
            ax.set_xlabel(inp.replace('_', ' '), fontsize=9)
        if j == 0:
            ax.set_ylabel(out.replace('_', ' '), fontsize=9)

        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(output_dir / 'eda_02_scatter_matrix.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("   Saved → eda_02_scatter_matrix.png")

# ── Plot 3: Output Distributions ─────────────────────────────
print("\n── Plot 3: Output Distributions ──")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Quality Target Distributions', fontsize=13)

for ax, col in zip(axes, OUTPUTS):
    data = df[col]
    ax.hist(data, bins=40, color='#5b8dd9',
            alpha=0.75, edgecolor='none', density=True)

    # Overlay normal curve
    mu, std = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, std),
            color='#e07b54', linewidth=2)

    ax.set_title(col.replace('_', ' ').title(), fontsize=10)
    ax.set_xlabel(f'mean={mu:.2f}  std={std:.2f}', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / 'eda_03_output_distributions.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("   Saved → eda_03_output_distributions.png")

# ── Plot 4: 3D Surface — Temperature × Residence Time → Purity
print("\n── Plot 4: 3D Surface Plot ──")

fig = plt.figure(figsize=(14, 5))
fig.suptitle('3D Response Surfaces — Non-linear Interactions', fontsize=13)

# Surface 1: temp × residence_time → purity
ax1 = fig.add_subplot(121, projection='3d')
temp_grid  = np.linspace(60, 120, 40)
rtime_grid = np.linspace(10, 60, 40)
T, R = np.meshgrid(temp_grid, rtime_grid)

P = (99
     - 0.002 * (T - 90) ** 2
     - 0.003 * (R - 35) ** 2
     - 0.001 * T * R * 0.1)

surf1 = ax1.plot_surface(T, R, P, cmap='RdYlGn',
                          alpha=0.85, edgecolor='none')
ax1.set_xlabel('Temperature (°C)', fontsize=9)
ax1.set_ylabel('Residence Time (min)', fontsize=9)
ax1.set_zlabel('Purity (%)', fontsize=9)
ax1.set_title('Temp × Residence Time → Purity', fontsize=10)
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Surface 2: temp × mixing_speed → viscosity
ax2 = fig.add_subplot(122, projection='3d')
speed_grid = np.linspace(100, 800, 40)
T2, S = np.meshgrid(temp_grid, speed_grid)

V = (800
     - 0.6  * S
     - 1.2  * T2
     + 0.004 * T2 * S * 0.1)

surf2 = ax2.plot_surface(T2, S, V, cmap='coolwarm',
                          alpha=0.85, edgecolor='none')
ax2.set_xlabel('Temperature (°C)', fontsize=9)
ax2.set_ylabel('Mixing Speed (RPM)', fontsize=9)
ax2.set_zlabel('Viscosity (cP)', fontsize=9)
ax2.set_title('Temp × Mixing Speed → Viscosity', fontsize=10)
fig.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.savefig(output_dir / 'eda_04_3d_surfaces.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("   Saved → eda_04_3d_surfaces.png")

# ── Plot 5: Outlier Detection (IQR method) ────────────────────
print("\n── Plot 5: Outlier Detection ──")

def flag_outliers_iqr(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)

outlier_flags = pd.DataFrame()
for col in OUTPUTS:
    outlier_flags[col] = flag_outliers_iqr(df[col])

df['is_outlier'] = outlier_flags.any(axis=1)
n_outliers = df['is_outlier'].sum()
print(f"   Outlier batches flagged: {n_outliers} "
      f"({n_outliers/len(df)*100:.1f}% of total)")

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Outlier Detection — Quality Targets (IQR method)', fontsize=13)

for ax, col in zip(axes, OUTPUTS):
    normal  = df[~df['is_outlier']][col]
    outlier = df[df['is_outlier']][col]
    ax.scatter(range(len(normal)), normal,
               alpha=0.2, s=5, color='#5b8dd9',
               label='Normal', edgecolors='none')
    ax.scatter(outlier.index, outlier,
               alpha=0.8, s=20, color='#e07b54',
               label='Outlier', edgecolors='none')
    ax.set_title(col.replace('_', ' ').title(), fontsize=10)
    ax.set_xlabel('Batch index', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'eda_05_outliers.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("   Saved → eda_05_outliers.png")

# ── EDA Summary ───────────────────────────────────────────────
print("\n" + "="*55)
print("EDA COMPLETE — Key findings")
print("="*55)
print(f"\nDataset: {df.shape[0]} batches × {df.shape[1]} columns")
print(f"Outlier batches: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")

print("\nTop correlations (inputs → outputs):")
for out in OUTPUTS:
    top = corr[out][INPUTS].abs().idxmax()
    val = corr[out][top]
    print(f"  {out:<20} ← strongest driver: "
          f"{top} (r={val:.2f})")

print("\n✅ All 5 EDA plots saved to data/")