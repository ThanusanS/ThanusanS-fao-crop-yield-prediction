"""
preprocess.py
Loads raw FAO FAOSTAT data, reshapes it into a flat
(area, item, year) -> yield table, and saves to processed/.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH  = os.path.join(os.path.dirname(__file__), "../data/raw/Production_Crops_Livestock_E_All_Data_NOFLAG.csv")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "../data/processed/crop_yield_clean.csv")

YEAR_COLS = [f"Y{y}" for y in range(1990, 2024)]   # 1990-2023
TOP_CROPS = [
    "Wheat", "Rice, paddy", "Maize (corn)", "Barley",
    "Soybeans", "Sugar cane", "Potatoes", "Cassava",
    "Cotton seed", "Sorghum", "Sunflower seed", "Rapeseed",
    "Groundnuts, excluding shelled", "Millet",
    "Bananas", "Tomatoes", "Onions and shallots, dry (excluding dehydrated)",
    "Beans, dry", "Lentils, dry", "Oranges"
]


def load_and_reshape():
    print("Loading raw FAO data …")
    df = pd.read_csv(RAW_PATH, encoding="latin1",
                     usecols=["Area", "Item", "Element"] + YEAR_COLS)

    # ── Separate the three elements ──────────────────────────────────────────
    yield_df  = df[df["Element"] == "Yield"][["Area","Item"] + YEAR_COLS].copy()
    area_df   = df[df["Element"] == "Area harvested"][["Area","Item"] + YEAR_COLS].copy()
    prod_df   = df[df["Element"] == "Production"][["Area","Item"] + YEAR_COLS].copy()

    def melt_element(frame, value_name):
        m = frame.melt(id_vars=["Area","Item"], value_vars=YEAR_COLS,
                       var_name="Year", value_name=value_name)
        m["Year"] = m["Year"].str.replace("Y","").astype(int)
        return m

    y  = melt_element(yield_df,  "yield_kg_ha")
    a  = melt_element(area_df,   "area_harvested_ha")
    p  = melt_element(prod_df,   "production_tonnes")

    merged = y.merge(a, on=["Area","Item","Year"], how="left") \
               .merge(p, on=["Area","Item","Year"], how="left")

    # ── Filter to top crops only ─────────────────────────────────────────────
    merged = merged[merged["Item"].isin(TOP_CROPS)]

    # ── Drop rows with no yield ──────────────────────────────────────────────
    merged = merged.dropna(subset=["yield_kg_ha"])
    merged = merged[merged["yield_kg_ha"] > 0]

    # ── Feature engineering ──────────────────────────────────────────────────
    # Encode crop and area as integers
    merged["crop_code"]  = pd.Categorical(merged["Item"]).codes
    merged["area_code"]  = pd.Categorical(merged["Area"]).codes

    # Decade feature
    merged["decade"] = (merged["Year"] // 10) * 10

    # Yield trend per crop×area: rolling 5-yr mean (lag so no leakage)
    merged = merged.sort_values(["Area","Item","Year"])
    merged["yield_lag1"] = merged.groupby(["Area","Item"])["yield_kg_ha"].shift(1)
    merged["yield_lag3"] = merged.groupby(["Area","Item"])["yield_kg_ha"].shift(3)
    merged["yield_roll5"] = (
        merged.groupby(["Area","Item"])["yield_kg_ha"]
              .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    )

    # Log-transform yield (right-skewed distribution)
    merged["log_yield"] = np.log1p(merged["yield_kg_ha"])

    # Year offset from 1990
    merged["year_offset"] = merged["Year"] - 1990

    # Production per area (proxy for intensity)
    merged["prod_per_ha"] = merged["production_tonnes"] / (merged["area_harvested_ha"] + 1)

    # Drop rows still missing after feature engineering
    merged = merged.dropna(subset=["yield_lag1","yield_lag3","yield_roll5"])
    merged = merged.reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(merged):,} rows → {OUT_PATH}")
    print(f"  Countries : {merged['Area'].nunique()}")
    print(f"  Crops     : {merged['Item'].nunique()}")
    print(f"  Year range: {merged['Year'].min()}–{merged['Year'].max()}")
    return merged


if __name__ == "__main__":
    load_and_reshape()
