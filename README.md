# 🏭 ML Quality Prediction — Process Manufacturing

A full end-to-end multi-output regression system that predicts
**moisture, viscosity, purity, and color deviation** from real-time
process conditions — replicating what process engineers do manually
in food, chemicals, and pharma manufacturing.

## 🎯 Project Overview

Built a complete ML quality prediction pipeline that takes 4 process
inputs (reactor temperature, mixing speed, residence time, feed
composition) and predicts 4 continuous quality targets simultaneously
using a tuned XGBoost model with SHAP explainability — replacing a
4-hour lab turnaround with a 2-second inference.

## 📊 Live App

Run locally with:

    streamlit run app/app.py

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Data Generation | Python, NumPy, Pandas |
| Feature Engineering | Physics-informed interaction terms |
| Modeling | Scikit-learn, XGBoost, MultiOutputRegressor |
| Hyperparameter Tuning | Optuna (50 trials) |
| Explainability | SHAP (TreeExplainer, waterfall plots) |
| Web App | Streamlit |
| Version Control | Git & GitHub |

## 📈 Key Results

- **3,000** synthetic production batches generated and processed
- **R² = 0.986** — viscosity prediction (near-perfect signal capture)
- **R² = 0.865** — moisture prediction
- **4 models** benchmarked — Linear Regression, Random Forest, XGBoost, MLP
- **Optuna tuning** — 50 trials to find best XGBoost hyperparameters
- **16 engineered features** — physics-informed interaction terms
- **All 4 KPI thresholds met** on held-out test set

## 🔍 Key Findings

- Purity peaks at **90–95°C** — above 100°C over-reaction degrades product
- Viscosity is dominated by mixing speed (r = −0.98) — near-linear relationship
- Purity non-linearity confirmed — Linear Regression fails, XGBoost captures sweet spot
- 42 outlier batches (1.4%) correctly flagged — matching injected failure rate
- Feed composition is the strongest driver of color deviation (r = −0.71)

## 📊 Model Performance

| Target | R² | RMSE | KPI | Status |
|---|---|---|---|---|
| Moisture | 0.865 | 0.616 % | ≤ 0.65 % | ✅ |
| Viscosity | 0.986 | 13.39 cP | ≤ 3% rel | ✅ |
| Purity | 0.656 | 0.546 % | ≤ 0.55 % | ✅ |
| Color deviation | 0.495 | 0.874 ΔE | ≤ 2.0 ΔE | ✅ |

## 💼 Business Impact

| Problem | Solution |
|---|---|
| Lab QC takes 4 hours per batch | Model predicts in < 2 seconds |
| Off-spec batches caught too late | Real-time prediction flags issues before release |
| Engineers cannot see input-output tradeoffs | Built-in what-if analysis sweeps any process variable |
| Black-box model engineers won't trust | SHAP waterfall explains every single prediction |
| One model per quality target = maintenance burden | Single multi-output model predicts all 4 targets simultaneously |

> **Bottom line:** Replaces a 4-hour lab turnaround with a 2-second inference —
> catching off-spec batches before they leave the reactor.

## 🏭 Industry Applications

| Industry | Target | Business Value |
|---|---|---|
| 🍞 Food manufacturing | Moisture | Controls shelf life prediction |
| 💊 Pharma | Purity | Drives regulatory compliance |
| ⚗️ Chemicals | Viscosity | Determines product grade |

## 📁 Project Structure

    quality-prediction-project/
    ├── data/
    │   ├── synthetic_process_data.csv
    │   ├── distribution_check.png
    │   ├── eda_01_correlation_heatmap.png
    │   ├── eda_02_scatter_matrix.png
    │   ├── eda_03_output_distributions.png
    │   ├── eda_04_3d_surfaces.png
    │   ├── eda_05_outliers.png
    │   ├── eval_01_residuals.png
    │   ├── eval_02_shap_*.png
    │   ├── eval_03_waterfall_*.png
    │   └── eval_04_whatif_temperature.png
    ├── notebooks/
    │   ├── 01_data_generation.py
    │   ├── 02_eda.py
    │   ├── 03_modeling.py
    │   ├── 04_tuning.py
    │   ├── 05_improved_model.py
    │   └── 06_shap_evaluation.py
    ├── models/
    │   ├── final_model.pkl
    │   ├── tuned_xgboost.pkl
    │   └── features.pkl
    ├── app/
    │   └── app.py
    ├── docs/
    │   ├── problem_statement.md
    │   └── kpi_targets.md
    └── README.md

## 🚀 How to Run

### 1. Create and activate environment

    conda create -n quality-prediction python=3.11 -y
    conda activate quality-prediction

### 2. Install dependencies

    pip install numpy pandas scikit-learn xgboost lightgbm shap plotly optuna streamlit joblib matplotlib seaborn scipy

### 3. Generate the dataset

    python notebooks/01_data_generation.py

### 4. Run EDA

    python notebooks/02_eda.py

### 5. Train and tune models

    python notebooks/03_modeling.py
    python notebooks/04_tuning.py
    python notebooks/05_improved_model.py

### 6. Run SHAP evaluation

    python notebooks/06_shap_evaluation.py

### 7. Launch the app

    streamlit run app/app.py

## 🤖 What-if Analysis Example

| Temp (°C) | Moisture (%) | Viscosity (cP) | Purity (%) | Color (ΔE) |
|---|---|---|---|---|
| 70 | 11.06 | 483.5 | 98.28 | 5.18 |
| 80 | 10.24 | 469.7 | 98.82 | 5.40 |
| **90** | **9.19** | **459.3** | **99.03** | **5.72** |
| 100 | 8.62 | 448.5 | 98.81 | 5.79 |
| 110 | 7.65 | 438.9 | 98.17 | 5.72 |
| 120 | 7.10 | 428.0 | 97.14 | 6.04 |

> Run at **90°C** for peak purity. Above 100°C purity degrades with no compensating benefit.