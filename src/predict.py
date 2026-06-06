"""
predict.py
Load trained XGBoost model and make yield predictions
for a given crop, country, and year.
"""

import joblib
import numpy as np
import pandas as pd
import os

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "../models")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/processed/crop_yield_clean.csv")


def load_model():
    model    = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    features = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
    return model, features


def get_lookup(df: pd.DataFrame):
    """Return crop→code and area→code mappings from training data."""
    crop_map = {c: i for i, c in enumerate(sorted(df["Item"].unique()))}
    area_map = {a: i for i, a in enumerate(sorted(df["Area"].unique()))}
    return crop_map, area_map


def predict_yield(crop: str, country: str, year: int,
                  area_ha: float = None,
                  prod_tonnes: float = None) -> dict:
    """
    Predict crop yield (kg/ha) for a given crop / country / year.

    Parameters
    ----------
    crop         : Crop name  (must match FAO item names)
    country      : Country name (must match FAO area names)
    year         : Target year
    area_ha      : Area harvested (ha) — uses historical median if None
    prod_tonnes  : Production (tonnes) — uses historical median if None

    Returns
    -------
    dict with predicted_yield_kg_ha and predicted_yield_t_ha
    """
    model, features = load_model()
    df = pd.read_csv(DATA_PATH)

    crop_map, area_map = get_lookup(df)

    if crop not in crop_map:
        raise ValueError(f"Crop '{crop}' not found. Available: {list(crop_map.keys())[:10]} …")
    if country not in area_map:
        raise ValueError(f"Country '{country}' not found.")

    # Historical stats for lag features
    hist = df[(df["Item"] == crop) & (df["Area"] == country)].sort_values("Year")

    if len(hist) < 3:
        raise ValueError(f"Not enough history for {crop} in {country}.")

    yield_lag1  = hist["yield_kg_ha"].iloc[-1]
    yield_lag3  = hist["yield_kg_ha"].iloc[-3]
    yield_roll5 = hist["yield_kg_ha"].tail(5).mean()
    area_med    = area_ha    if area_ha    else hist["area_harvested_ha"].median()
    prod_med    = prod_tonnes if prod_tonnes else hist["production_tonnes"].median()
    prod_per_ha = prod_med / (area_med + 1)

    row = {
        "crop_code"       : crop_map[crop],
        "area_code"       : area_map[country],
        "year_offset"     : year - 1990,
        "decade"          : (year // 10) * 10,
        "yield_lag1"      : yield_lag1,
        "yield_lag3"      : yield_lag3,
        "yield_roll5"     : yield_roll5,
        "area_harvested_ha": area_med,
        "prod_per_ha"     : prod_per_ha,
    }

    X = pd.DataFrame([row])[features]
    pred = float(model.predict(X)[0])

    return {
        "crop"                  : crop,
        "country"               : country,
        "year"                  : year,
        "predicted_yield_kg_ha" : round(pred, 2),
        "predicted_yield_t_ha"  : round(pred / 1000, 3),
    }


def batch_predict(requests: list[dict]) -> pd.DataFrame:
    """Run predict_yield for a list of dicts."""
    results = []
    for req in requests:
        try:
            results.append(predict_yield(**req))
        except Exception as e:
            results.append({"error": str(e), **req})
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Quick smoke test
    result = predict_yield("Wheat", "India", 2025)
    print(result)

    result2 = predict_yield("Rice, paddy", "Sri Lanka", 2025)
    print(result2)

    result3 = predict_yield("Maize (corn)", "United States of America", 2025)
    print(result3)
