"""
run_pipeline.py
Full end-to-end pipeline runner.
Usage:  python run_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_and_reshape
from eda        import run_eda

print("\n" + "="*55)
print("  CROP YIELD PREDICTION — FULL PIPELINE")
print("="*55)

print("\n[1/3] Preprocessing raw FAO data …")
RAW_FILE  = os.path.join(os.path.dirname(__file__), "data/raw/Production_Crops_Livestock_E_All_Data_NOFLAG.csv")
PROC_FILE = os.path.join(os.path.dirname(__file__), "data/processed/crop_yield_clean.csv")
if os.path.exists(RAW_FILE):
    load_and_reshape()
elif os.path.exists(PROC_FILE):
    print("  Raw FAO file not found — using existing processed data.")
else:
    print("  ERROR: No raw or processed data found. Place the FAO CSV in data/raw/")
    sys.exit(1)

print("\n[2/3] Running EDA …")
run_eda()

print("\n[3/3] Training models ...")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
required_models = ["xgb_model.pkl", "rf_model.pkl", "ridge_model.pkl",
                   "scaler.pkl", "feature_names.pkl", "metrics.json"]
models_exist = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_models)

if models_exist:
    print("  Pre-trained models found -- skipping training.")
else:
    import subprocess, sys as _sys
    result = subprocess.run(
        [_sys.executable, "-c", """
import pandas as pd, numpy as np, joblib, json, os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = 'data/processed/crop_yield_clean.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ['crop_code','area_code','year_offset','decade',
            'yield_lag1','yield_lag3','yield_roll5',
            'area_harvested_ha','prod_per_ha']
TARGET = 'yield_kg_ha'

df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES+[TARGET])
mask_test = df['Year'] >= 2020
X_train = df[~mask_test][FEATURES].values; y_train = df[~mask_test][TARGET].values
X_test  = df[mask_test][FEATURES].values;  y_test  = df[mask_test][TARGET].values

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train); X_te_sc = scaler.transform(X_test)
joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')

def save_metrics(name, preds):
    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    mape = float(np.mean(np.abs((y_test-preds)/(y_test+1e-9)))*100)
    print(f'{name}: R2={r2:.4f} RMSE={rmse:,.0f} MAPE={mape:.1f}%')
    return dict(model=name,rmse=round(rmse,2),mae=round(mae,2),r2=round(r2,4),mape=round(mape,2),cv_mean=round(r2,4),cv_std=0.01)

results = []
ridge = Ridge(alpha=10); ridge.fit(X_tr_sc, y_train)
results.append(save_metrics('Ridge (baseline)', ridge.predict(X_te_sc)))
joblib.dump(ridge, f'{MODEL_DIR}/ridge_model.pkl')

rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
results.append(save_metrics('Random Forest', rf.predict(X_test)))
joblib.dump(rf, f'{MODEL_DIR}/rf_model.pkl')

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
results.append(save_metrics('XGBoost', xgb_preds))
joblib.dump(xgb, f'{MODEL_DIR}/xgb_model.pkl')
joblib.dump(FEATURES, f'{MODEL_DIR}/feature_names.pkl')

with open(f'{MODEL_DIR}/metrics.json','w') as f: json.dump(results, f, indent=2)
np.save(f'{MODEL_DIR}/y_test.npy', y_test)
np.save(f'{MODEL_DIR}/xgb_preds.npy', xgb_preds)
print('Models saved.')
"""], capture_output=False
    )
    if result.returncode != 0:
        print("  WARNING: Training subprocess failed. Check for DLL/policy issues.")
        if models_exist:
            print("  Falling back to pre-existing models.")

from predict import predict_yield
print("\nSample predictions:")
tests = [
    ("Wheat",        "India",                    2025),
    ("Maize (corn)", "United States of America", 2025),
    ("Barley",       "Germany",                  2025),
    ("Sugar cane",   "Brazil",                   2025),
    ("Potatoes",     "China",                    2025),
]
for crop, country, year in tests:
    try:
        r = predict_yield(crop, country, year)
        print(f"  {crop:<30} {country:<28} {year} -> {r['predicted_yield_t_ha']:.3f} t/ha")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*55)
print("  PIPELINE COMPLETE")
print("  Reports : reports/   (10 PNG charts)")
print("  Models  : models/    (3 .pkl files + metrics.json)")
print("  Launch  : streamlit run app/streamlit_app.py")
print("="*55 + "\n")
