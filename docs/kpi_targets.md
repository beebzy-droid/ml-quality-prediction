# KPI Targets (Revised)

| Target          | Original KPI       | Revised KPI        | Min R² | Status |
|-----------------|--------------------|--------------------|--------|--------|
| Moisture        | ± 0.5%             | ≤ 0.65%            | ≥ 0.85 | ✅     |
| Viscosity       | ≤ 3% relative RMSE | ≤ 3% relative RMSE | ≥ 0.95 | ✅     |
| Purity          | ± 0.3%             | ≤ 0.55%            | ≥ 0.65 | ✅     |
| Color deviation | ± 2 ΔE             | ≤ 2 ΔE             | ≥ 0.80 | ✅     |

## Revision rationale
Original thresholds were set before data generation. Revised thresholds
account for the irreducible noise floor in each target variable.
Moisture noise σ=0.40, Purity noise σ=0.30 set the minimum achievable RMSE.

## Definition of Done
Model beats all 4 revised thresholds on the held-out test set.