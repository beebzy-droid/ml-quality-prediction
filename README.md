# 🏭 ML Quality Prediction — Process Manufacturing

Predict **moisture, viscosity, purity, and color deviation** from 
real-time process conditions using a tuned XGBoost multi-output 
regression model.

> Replaces 4-hour lab turnaround with a **2-second inference.**

---

## Business Impact

| Problem | Solution |
|---|---|
| Lab QC takes 4 hours per batch | Model predicts in < 2 seconds |
| Off-spec batches caught too late | Real-time prediction flags issues early |
| Engineers cannot see input-output tradeoffs | What-if analysis built into the app |

---

## Live App

Run locally with:

    streamlit run app/app.py

---

## Model Performance

| Target | R² | RMSE | KPI | Status |
|---|---|---|---|---|
| Moisture | 0.865 | 0.616 % | ≤ 0.65 % | ✅ |
| Viscosity | 0.986 | 13.39 cP | ≤ 3% rel | ✅ |
| Purity | 0.656 | 0.546 % | ≤ 0.55 % | ✅ |
| Color deviation | 0.495 | 0.874 ΔE | ≤ 2.0 ΔE | ✅ |

---

## Key Engineering Finding

Run the reactor at 90–95°C. Purity peaks at 99%, moisture sits at
9.0–9.2%, and viscosity holds at 450–460 cP. Above 100°C purity
degrades with no compensating benefit.

---

## Project Structure

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
    │   ├── eval_02_shap_moisture.png
    │   ├── eval_02_shap_viscosity.png
    │   ├── eval_02_shap_purity.png
    │   ├── eval_02_shap_color_deviation.png
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

---

## Tech Stack

- Python 3.11
- XGBoost + scikit-learn — modeling
- SHAP — explainability
- Optuna — hyperparameter tuning
- Streamlit — interactive app
- Pandas + NumPy — data engineering
- Matplotlib + Seaborn — visualization

---

## Phases Completed

| Phase | Description | Deliverable |
|---|---|---|
| 1 | Problem definition | Problem statement + KPIs |
| 2 | Synthetic data generation | 3000-batch physics dataset |
| 3 | Exploratory data analysis | 5 visualization plots |
| 4 | Feature engineering + modeling | 4 models + Optuna tuning |
| 5 | Evaluation + SHAP | 10 evaluation plots + what-if |
| 6 | Streamlit app | Live interactive tool |

---

## Industry Applications

- 🍞 **Food manufacturing** — moisture controls shelf life
- 💊 **Pharma** — purity drives regulatory compliance
- ⚗️ **Chemicals** — viscosity determines product grade

---

## Dataset

Synthetic dataset of 3000 production batches generated using
physics-informed equations with:
- Non-linear input interactions (temp x residence time → purity)
- Equipment failure batches injected (1%)
- Seasonal feed composition shift (last 10% of batches)
- Gaussian sensor noise on all targets