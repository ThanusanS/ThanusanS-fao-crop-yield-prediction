"""
app/streamlit_app.py
Interactive crop yield prediction dashboard.
Run with:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, json, os, sys

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import predict_yield, get_lookup

DATA_PATH   = os.path.join(ROOT, "data/processed/crop_yield_clean.csv")
MODEL_DIR   = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; font-weight: 500; }
    [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 700; color: #3B6D11; }
    @media (max-width: 768px) {
        [data-testid="stMetricValue"] { font-size: 20px !important; }
        h1 { font-size: 24px !important; }
        h2 { font-size: 20px !important; }
        h3 { font-size: 16px !important; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_assets():
    model    = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    features = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
    return model, features


@st.cache_data
def load_metrics():
    with open(f"{MODEL_DIR}/metrics.json") as f:
        return json.load(f)


# ── Load assets ───────────────────────────────────────────────────────────────
df      = load_data()
model, features = load_model_assets()
crop_map, area_map = get_lookup(df)

crops     = sorted(crop_map.keys())
countries = sorted(area_map.keys())
metrics   = load_metrics()
xgb_m     = next(m for m in metrics if m["model"] == "XGBoost")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/FAO_logo.svg/200px-FAO_logo.svg.png", width=80)
    st.title("🌾 Crop Yield Predictor")
    st.caption("Powered by FAO FAOSTAT data · XGBoost model")
    st.divider()

    st.subheader("🔧 Prediction inputs")
    sel_crop    = st.selectbox("Crop", crops,
                                index=crops.index("Wheat") if "Wheat" in crops else 0)
    sel_country = st.selectbox("Country", countries,
                                index=countries.index("India") if "India" in countries else 0)
    sel_year    = st.slider("Target year", 2024, 2035, 2025)
    predict_btn = st.button("🚀 Predict yield", use_container_width=True, type="primary")
    st.divider()
    st.caption("Model: XGBoost  |  Data: FAO FAOSTAT 1990–2023")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🌾 Crop Yield Prediction Dashboard")
st.caption("FAO FAOSTAT · 244 countries · 20 major crops · 1990–2023")

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Model R²",    f"{xgb_m['r2']:.4f}")
k2.metric("RMSE (kg/ha)", f"{xgb_m['rmse']:,.0f}")
k3.metric("MAPE",        f"{xgb_m['mape']:.1f}%")
k4.metric("CV R²",       f"{xgb_m['cv_mean']:.4f} ± {xgb_m['cv_std']:.4f}")

st.divider()

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([1.4, 1])

with left:
    st.subheader("📈 Historical yield trend")

    hist = df[(df["Item"] == sel_crop) & (df["Area"] == sel_country)].sort_values("Year")

    if len(hist) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hist["Year"], hist["yield_kg_ha"] / 1000,
                color="#3B6D11", linewidth=2.2, marker="o", markersize=3)
        ax.fill_between(hist["Year"], hist["yield_kg_ha"] / 1000,
                        alpha=0.12, color="#3B6D11")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield (t/ha)")
        ax.set_title(f"{sel_crop} — {sel_country}")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("No historical data found for this crop/country combination.")

    # EDA plots
    st.subheader("📊 EDA — global insights")
    report_imgs = {
        "Yield distribution" : "01_yield_distribution.png",
        "Yield trends"       : "02_yield_trends.png",
        "Yield by crop"      : "03_yield_by_crop.png",
        "Correlation heatmap": "04_correlation_heatmap.png",
        "Top countries"      : "05_top_countries.png",
        "Model comparison"   : "06_model_comparison.png",
        "Actual vs Predicted": "07_actual_vs_predicted.png",
        "SHAP importance"    : "09_shap_importance.png",
        "SHAP beeswarm"      : "10_shap_beeswarm.png",
    }
    sel_plot = st.selectbox("Choose plot", list(report_imgs.keys()))
    img_path = os.path.join(REPORTS_DIR, report_imgs[sel_plot])
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.info("Run training pipeline first to generate plots.")

with right:
    st.subheader("🎯 Prediction result")

    if predict_btn:
        with st.spinner("Predicting …"):
            try:
                result = predict_yield(sel_crop, sel_country, sel_year)
                st.success("Prediction complete!")
                st.metric(
                    label=f"Predicted yield — {sel_year}",
                    value=f"{result['predicted_yield_t_ha']:.3f} t/ha",
                    delta=None
                )
                st.metric("kg/ha", f"{result['predicted_yield_kg_ha']:,.0f}")

                # Compare with last known value
                if len(hist) > 0:
                    last_val = hist["yield_kg_ha"].iloc[-1]
                    diff_pct = (result["predicted_yield_kg_ha"] - last_val) / last_val * 100
                    st.metric(
                        "vs last known year",
                        f"{last_val/1000:.3f} t/ha",
                        delta=f"{diff_pct:+.1f}%"
                    )
            except Exception as e:
                st.error(str(e))
    else:
        st.info("👈 Select a crop, country, and year then click **Predict yield**.")

    # ── Model metrics table ───────────────────────────────────────────────────
    st.subheader("📋 Model comparison")
    metrics_df = pd.DataFrame(metrics)[["model","r2","rmse","mae","mape","cv_mean"]]
    metrics_df.columns = ["Model","R²","RMSE","MAE","MAPE%","CV R²"]
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    # ── Data explorer ─────────────────────────────────────────────────────────
    st.subheader("🗂️ Data explorer")
    crop_filter = st.selectbox("Filter by crop", ["All"] + crops, key="de_crop")
    sub = df if crop_filter == "All" else df[df["Item"] == crop_filter]
    latest_year = sub["Year"].max()
    show = (sub[sub["Year"] == latest_year]
              [["Area","Item","Year","yield_kg_ha","area_harvested_ha","production_tonnes"]]
              .rename(columns={"yield_kg_ha":"Yield kg/ha",
                               "area_harvested_ha":"Area ha",
                               "production_tonnes":"Production t"})
              .sort_values("Yield kg/ha", ascending=False)
              .head(30))
    st.dataframe(show.reset_index(drop=True), use_container_width=True)

st.divider()
st.caption("Data source: FAO FAOSTAT · Model: XGBoost · Built with Streamlit")
