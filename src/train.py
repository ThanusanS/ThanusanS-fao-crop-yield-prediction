"""
train.py
Trains XGBoost and Random Forest models on the preprocessed FAO yield data.
Evaluates on held-out test set, generates SHAP plots, and saves models.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, json

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.ensemble           import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model       import Ridge
from sklearn.metrics            import (mean_squared_error,
                                        mean_absolute_error, r2_score)
from xgboost import XGBRegressor
import shap

DATA_PATH    = os.path.join(os.path.dirname(__file__), "../data/processed/crop_yield_clean.csv")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "../models")
REPORTS_DIR  = os.path.join(os.path.dirname(__file__), "../reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

FEATURES = [
    "crop_code", "area_code", "year_offset", "decade",
    "yield_lag1", "yield_lag3", "yield_roll5",
    "area_harvested_ha", "prod_per_ha"
]
TARGET = "yield_kg_ha"


# ── Evaluation helper ────────────────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    preds = model.predict(X_te)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    mae   = mean_absolute_error(y_te, preds)
    r2    = r2_score(y_te, preds)
    mape  = float(np.mean(np.abs((y_te - preds) / (y_te + 1e-9))) * 100)
    cv    = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2", n_jobs=-1)
    result = dict(model=name, rmse=round(rmse,2), mae=round(mae,2),
                  r2=round(r2,4), mape=round(mape,2),
                  cv_mean=round(cv.mean(),4), cv_std=round(cv.std(),4))
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"  RMSE : {rmse:>10,.2f} kg/ha")
    print(f"  MAE  : {mae:>10,.2f} kg/ha")
    print(f"  R²   : {r2:>10.4f}")
    print(f"  MAPE : {mape:>10.2f}%")
    print(f"  CV R²: {cv.mean():.4f} ± {cv.std():.4f}")
    return result, preds


def run_training():
    print("Loading processed data …")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"  {len(df):,} rows, {df['Item'].nunique()} crops, {df['Area'].nunique()} countries")

    X = df[FEATURES].values
    y = df[TARGET].values

    # ── Train / test split (time-based: test = last 3 years) ─────────────────
    mask_test  = df["Year"] >= 2020
    mask_train = ~mask_test
    X_train, y_train = X[mask_train], y[mask_train]
    X_test,  y_test  = X[mask_test],  y[mask_test]
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Scale (needed for Ridge baseline) ────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    results = []

    # ── 1. Ridge baseline ────────────────────────────────────────────────────
    ridge = Ridge(alpha=10)
    ridge.fit(X_train_sc, y_train)
    r, _ = evaluate("Ridge (baseline)", ridge, X_train_sc, y_train, X_test_sc, y_test)
    results.append(r)
    joblib.dump(ridge, f"{MODEL_DIR}/ridge_model.pkl")

    # ── 2. Random Forest ─────────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=300, max_depth=12,
                               min_samples_leaf=3, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    r, rf_preds = evaluate("Random Forest", rf, X_train, y_train, X_test, y_test)
    results.append(r)
    joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")

    # ── 3. XGBoost (best) ────────────────────────────────────────────────────
    xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, verbosity=0, n_jobs=-1
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    r, xgb_preds = evaluate("XGBoost", xgb, X_train, y_train, X_test, y_test)
    results.append(r)
    joblib.dump(xgb, f"{MODEL_DIR}/xgb_model.pkl")

    # ── Save feature names alongside models ──────────────────────────────────
    joblib.dump(FEATURES, f"{MODEL_DIR}/feature_names.pkl")

    # ── Save metrics to JSON ──────────────────────────────────────────────────
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot: model comparison bar chart ────────────────────────────────────
    res_df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    colors = ["#B4B2A9","#97C459","#3B6D11"]
    for ax, metric, label in zip(axes,
                                  ["r2","rmse","mape"],
                                  ["R² (higher is better)",
                                   "RMSE kg/ha (lower is better)",
                                   "MAPE % (lower is better)"]):
        bars = ax.bar(res_df["model"], res_df[metric], color=colors, alpha=0.9)
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h*1.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("Model comparison — FAO crop yield prediction", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/06_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved 06_model_comparison.png")

    # ── Plot: actual vs predicted (XGBoost) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test/1000, xgb_preds/1000, alpha=0.25, s=8,
               color="#3B6D11", label="XGBoost")
    lim = [0, max(y_test.max(), xgb_preds.max())/1000]
    ax.plot(lim, lim, "k--", lw=1, label="Perfect prediction")
    ax.set_xlabel("Actual yield (t/ha)", fontsize=12)
    ax.set_ylabel("Predicted yield (t/ha)", fontsize=12)
    ax.set_title("Actual vs Predicted yield — XGBoost", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/07_actual_vs_predicted.png", dpi=150)
    plt.close()
    print("  Saved 07_actual_vs_predicted.png")

    # ── Plot: residuals ───────────────────────────────────────────────────────
    residuals = y_test - xgb_preds
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xgb_preds/1000, residuals/1000, alpha=0.2, s=8, color="#378ADD")
    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.set_xlabel("Predicted yield (t/ha)", fontsize=12)
    ax.set_ylabel("Residual (t/ha)", fontsize=12)
    ax.set_title("Residual plot — XGBoost", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/08_residuals.png", dpi=150)
    plt.close()
    print("  Saved 08_residuals.png")

    # ── SHAP feature importance ──────────────────────────────────────────────
    print("\nComputing SHAP values …")
    sample_idx = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]

    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=FEATURES,
                      plot_type="bar", show=False)
    plt.title("SHAP feature importance — XGBoost", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/09_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 09_shap_importance.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=FEATURES, show=False)
    plt.title("SHAP beeswarm — XGBoost", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/10_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 10_shap_beeswarm.png")

    print("\nTraining complete. All models saved to models/")
    return results


if __name__ == "__main__":
    run_training()
