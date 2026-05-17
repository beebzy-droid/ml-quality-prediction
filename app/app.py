# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:25:57 2026

@author: Bien
"""

# =============================================================
# Phase 6 — Streamlit App
# ML Quality Prediction — Process Manufacturing
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Quality Prediction — Process Manufacturing",
    page_icon="🏭",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
model_dir = BASE / 'models'

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    saved   = joblib.load(model_dir / 'final_model.pkl')
    return saved['model'], saved['scaler'], saved['features']

model, scaler, FEATURES = load_model()

OUTPUTS    = ['moisture', 'viscosity', 'purity', 'color_deviation']
UNITS      = {'moisture': '%', 'viscosity': 'cP',
              'purity': '%', 'color_deviation': 'ΔE'}
KPI_LIMITS = {'moisture': (0, 12), 'viscosity': (200, 700),
              'purity': (95, 100), 'color_deviation': (0, 8)}
KPI_GOOD   = {'moisture': (4, 11), 'viscosity': (300, 600),
              'purity': (97, 100), 'color_deviation': (0, 5)}

# ── Feature builder ───────────────────────────────────────────
def build_features(temp, speed, rtime, comp):
    row = {
        'temperature'         : temp,
        'mixing_speed'        : speed,
        'residence_time'      : rtime,
        'feed_composition'    : comp,
        'temp_x_residence'    : temp  * rtime,
        'temp_x_speed'        : temp  * speed,
        'speed_x_composition' : speed * comp,
        'temp_squared'        : temp  ** 2,
        'residence_squared'   : rtime ** 2,
        'temp_centered'       : (temp  - 90) ** 2,
        'residence_centered'  : (rtime - 35) ** 2,
        'temp_res_interaction': temp  * rtime * 0.1,
        'temp_x_composition'  : temp  * comp,
        'temp_cubed_norm'     : (temp / 100) ** 3,
        'composition_squared' : comp  ** 2,
        'comp_x_temp'         : comp  * temp,
    }
    return pd.DataFrame([row])[FEATURES]

def predict(temp, speed, rtime, comp):
    X   = build_features(temp, speed, rtime, comp)
    Xsc = scaler.transform(X)
    return model.predict(Xsc)[0], Xsc

def traffic_light(val, col):
    lo, hi = KPI_GOOD[col]
    if lo <= val <= hi:
        return '🟢'
    elif val < KPI_LIMITS[col][0] or val > KPI_LIMITS[col][1]:
        return '🔴'
    return '🟡'

# ── Header ────────────────────────────────────────────────────
st.title("🏭 ML Quality Prediction — Process Manufacturing")
st.markdown(
    "Predict **moisture, viscosity, purity, and color deviation** "
    "from real-time process conditions. "
    "Replaces 4-hour lab turnaround with a **2-second inference.**"
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎛️ Predict", "📈 What-if Analysis",
    "🔍 SHAP Explanation", "📊 Model Info"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Predict
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Set process conditions")
    st.markdown("Adjust the sliders and predictions update instantly.")

    c1, c2 = st.columns(2)
    with c1:
        temp  = st.slider("🌡️ Reactor temperature (°C)",
                          60.0, 120.0, 90.0, 0.5)
        speed = st.slider("⚙️ Mixing speed (RPM)",
                          100.0, 800.0, 450.0, 5.0)
    with c2:
        rtime = st.slider("⏱️ Residence time (min)",
                          10.0, 60.0, 35.0, 0.5)
        comp  = st.slider("🧪 Feed composition (fraction)",
                          0.40, 0.90, 0.65, 0.01)

    preds, _ = predict(temp, speed, rtime, comp)

    st.divider()
    st.subheader("Predicted quality outputs")

    cols = st.columns(4)
    for col_ui, out, pred in zip(cols, OUTPUTS, preds):
        light  = traffic_light(pred, out)
        lo, hi = KPI_GOOD[out]
        with col_ui:
            st.metric(
                label=f"{light} {out.replace('_',' ').title()}",
                value=f"{pred:.3f} {UNITS[out]}"
            )
            st.caption(f"Target range: {lo}–{hi} {UNITS[out]}")

    st.divider()
    st.markdown(
        "🟢 Within spec &nbsp;&nbsp; 🟡 Borderline &nbsp;&nbsp; "
        "🔴 Out of spec"
    )

# ════════════════════════════════════════════════════════════
# TAB 2 — What-if Analysis
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("What-if analysis")
    st.markdown(
        "Sweep one process input across its full range "
        "while holding others at their baseline values."
    )

    sweep_input = st.selectbox(
        "Select input to sweep:",
        ['temperature', 'mixing_speed',
         'residence_time', 'feed_composition']
    )

    defaults = {'temperature': 90.0, 'mixing_speed': 450.0,
                'residence_time': 35.0, 'feed_composition': 0.65}
    ranges   = {'temperature': (60, 120),
                'mixing_speed': (100, 800),
                'residence_time': (10, 60),
                'feed_composition': (0.40, 0.90)}
    units_in = {'temperature': '°C', 'mixing_speed': 'RPM',
                'residence_time': 'min', 'feed_composition': ''}

    lo_in, hi_in = ranges[sweep_input]
    sweep_vals   = np.linspace(lo_in, hi_in, 60)

    results = {out: [] for out in OUTPUTS}
    for v in sweep_vals:
        kwargs = dict(defaults)
        kwargs[sweep_input] = v
        p, _ = predict(
            temp  = kwargs['temperature'],
            speed = kwargs['mixing_speed'],
            rtime = kwargs['residence_time'],
            comp  = kwargs['feed_composition']
        )
        for j, out in enumerate(OUTPUTS):
            results[out].append(p[j])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f'Effect of {sweep_input.replace("_"," ").title()} '
        f'on Quality Targets', fontsize=12)

    for ax, out in zip(axes, OUTPUTS):
        ax.plot(sweep_vals, results[out],
                color='#5b8dd9', linewidth=2.5)
        ax.axvline(defaults[sweep_input], color='#e07b54',
                   linestyle='--', linewidth=1.5,
                   label=f'Default {defaults[sweep_input]}')
        lo_g, hi_g = KPI_GOOD[out]
        ax.axhspan(lo_g, hi_g, alpha=0.08, color='green',
                   label='Target range')
        ax.set_xlabel(
            f'{sweep_input.replace("_"," ").title()} '
            f'({units_in[sweep_input]})', fontsize=9)
        ax.set_ylabel(
            f'{out.replace("_"," ").title()} ({UNITS[out]})',
            fontsize=9)
        ax.set_title(out.replace('_', ' ').title(), fontsize=10)
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════
# TAB 3 — SHAP Explanation
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Why did the model predict this?")
    st.markdown(
        "SHAP waterfall chart for the current slider values. "
        "Shows which process conditions pushed the prediction "
        "up or down from the baseline."
    )

    target_choice = st.selectbox(
        "Select quality target to explain:",
        OUTPUTS,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    if st.button("Generate SHAP explanation"):
        with st.spinner("Computing SHAP values..."):
            _, X_sc = predict(temp, speed, rtime, comp)
            idx     = OUTPUTS.index(target_choice)
            est     = model.estimators_[idx]
            exp     = shap.TreeExplainer(est)
            sv      = exp(pd.DataFrame(
                X_sc, columns=FEATURES))

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            shap.waterfall_plot(sv[0], show=False)
            plt.title(
                f'SHAP — {target_choice.replace("_"," ").title()} '
                f'prediction explanation', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.info(
            "Red bars push the prediction **higher**. "
            "Blue bars push it **lower**. "
            "Length = magnitude of impact."
        )

# ════════════════════════════════════════════════════════════
# TAB 4 — Model Info
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model performance")

    perf = {
        'Target'    : ['Moisture', 'Viscosity',
                       'Purity', 'Color deviation'],
        'R²'        : [0.8651, 0.9859, 0.6555, 0.4954],
        'RMSE'      : ['0.616 %', '13.39 cP',
                       '0.546 %', '0.874 ΔE'],
        'KPI'       : ['≤ 0.65 %', '≤ 3% rel',
                       '≤ 0.55 %', '≤ 2.0 ΔE'],
        'Status'    : ['✅', '✅', '✅', '✅'],
    }
    st.dataframe(pd.DataFrame(perf), use_container_width=True,
                 hide_index=True)

    st.divider()
    st.subheader("Project summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Training batches", "2,700")
    col2.metric("Features engineered", "16")
    col3.metric("Inference time", "< 2 sec")

    st.divider()
    st.subheader("Industry applications")

    ia1, ia2, ia3 = st.columns(3)
    with ia1:
        st.markdown("**🍞 Food manufacturing**")
        st.markdown("Moisture → shelf life prediction")
    with ia2:
        st.markdown("**💊 Pharma**")
        st.markdown("Purity → regulatory compliance")
    with ia3:
        st.markdown("**⚗️ Chemicals**")
        st.markdown("Viscosity → product grade")