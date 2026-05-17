# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:09:22 2026

@author: Bien
"""

# =============================================================
# Phase 2 — Synthetic Data Generation
# Physics-informed process manufacturing dataset
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────
np.random.seed(42)
N = 3000  # number of batches to simulate

# ── 1. SIMULATE PROCESS INPUTS ───────────────────────────────
# Each input is drawn from a realistic operating range

temperature      = np.random.uniform(60, 120, N)   # °C
mixing_speed     = np.random.uniform(100, 800, N)  # RPM
residence_time   = np.random.uniform(10, 60, N)    # minutes
feed_composition = np.random.uniform(0.4, 0.9, N)  # fraction (0–1)

# ── 2. SIMULATE QUALITY OUTPUTS (physics-informed) ───────────

# -- Moisture (%) --
# Higher temp and longer time both drive moisture down
# Noise simulates sensor drift
moisture = (
    18
    - 0.08 * temperature
    - 0.05 * residence_time
    + 0.03 * feed_composition * 10
    + np.random.normal(0, 0.4, N)
)

# -- Viscosity (cP) --
# Higher mixing speed thins the product (shear thinning)
# Higher temp also reduces viscosity
# Non-linear: temp × mixing_speed interaction
viscosity = (
    800
    - 0.6  * mixing_speed
    - 1.2  * temperature
    + 0.004 * temperature * mixing_speed * 0.1
    + 20   * feed_composition
    + np.random.normal(0, 8, N)
)

# -- Purity (%) --
# Sweet spot: moderate temp + moderate residence time = high purity
# Over-reaction (high temp × long time) degrades purity
# Non-linear interaction: temp × residence_time
purity = (
    99
    - 0.002 * (temperature  - 90) ** 2
    - 0.003 * (residence_time - 35) ** 2
    - 0.001 * temperature * residence_time * 0.1
    + 0.5   * feed_composition
    + np.random.normal(0, 0.3, N)
)

# -- Color deviation (ΔE) --
# Feed composition is the dominant driver
# Temperature has a secondary effect
# Higher deviation = worse color quality
color_deviation = (
    2
    + 6   * (1 - feed_composition)
    + 0.02 * temperature
    - 0.01 * mixing_speed * 0.1
    + np.random.normal(0, 0.5, N)
)

# ── 3. INJECT DOMAIN REALISM ─────────────────────────────────

# -- Equipment failure batches (1% of batches) --
failure_idx = np.random.choice(N, size=int(0.01 * N), replace=False)
moisture[failure_idx]        += np.random.uniform(2, 5, len(failure_idx))
viscosity[failure_idx]       += np.random.uniform(50, 150, len(failure_idx))
purity[failure_idx]          -= np.random.uniform(2, 5, len(failure_idx))
color_deviation[failure_idx] += np.random.uniform(3, 8, len(failure_idx))

# -- Seasonal feed composition shift (last 10% of batches) --
seasonal_idx = np.arange(int(0.9 * N), N)
feed_composition[seasonal_idx] -= 0.08

# -- Clip to physically realistic bounds --
moisture         = np.clip(moisture, 0, 20)
viscosity        = np.clip(viscosity, 50, 1000)
purity           = np.clip(purity, 80, 100)
color_deviation  = np.clip(color_deviation, 0, 15)

# ── 4. ASSEMBLE DATAFRAME ────────────────────────────────────
df = pd.DataFrame({
    'temperature'      : np.round(temperature, 2),
    'mixing_speed'     : np.round(mixing_speed, 2),
    'residence_time'   : np.round(residence_time, 2),
    'feed_composition' : np.round(feed_composition, 3),
    'moisture'         : np.round(moisture, 3),
    'viscosity'        : np.round(viscosity, 2),
    'purity'           : np.round(purity, 3),
    'color_deviation'  : np.round(color_deviation, 3),
})

# ── 5. SAVE TO CSV ───────────────────────────────────────────
output_path = Path(__file__).parent.parent / 'data' / 'synthetic_process_data.csv'
df.to_csv(output_path, index=False)
print(f"✅ Dataset saved → {output_path}")
print(f"   Shape: {df.shape}")

# ── 6. QUICK VALIDATION ──────────────────────────────────────
print("\n── Input ranges ──────────────────────────────")
print(df[['temperature','mixing_speed',
          'residence_time','feed_composition']].describe().round(2))

print("\n── Output ranges ─────────────────────────────")
print(df[['moisture','viscosity',
          'purity','color_deviation']].describe().round(2))

# ── 7. QUICK VISUAL CHECK ────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle('Synthetic Process Manufacturing Dataset — Distribution Check',
             fontsize=13, y=1.01)

cols = ['temperature','mixing_speed','residence_time','feed_composition',
        'moisture','viscosity','purity','color_deviation']
colors = ['#5b8dd9','#5b8dd9','#5b8dd9','#5b8dd9',
          '#e07b54','#e07b54','#e07b54','#e07b54']

for ax, col, color in zip(axes.flatten(), cols, colors):
    ax.hist(df[col], bins=40, color=color, alpha=0.8, edgecolor='none')
    ax.set_title(col.replace('_', ' ').title(), fontsize=10)
    ax.set_ylabel('Count')
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(
    Path(__file__).parent.parent / 'data' / 'distribution_check.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print("\n✅ Distribution plot saved → data/distribution_check.png")