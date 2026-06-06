"""
eda.py
Generates exploratory data analysis plots saved to reports/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

DATA_PATH    = os.path.join(os.path.dirname(__file__), "../data/processed/crop_yield_clean.csv")
REPORTS_DIR  = os.path.join(os.path.dirname(__file__), "../reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

PALETTE = "#3B6D11"
sns.set_theme(style="whitegrid", font_scale=1.0)


def run_eda():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows for EDA")

    years = sorted(df["Year"].unique())
    crops = df["Item"].value_counts().head(10).index.tolist()

    # ── 1. Yield distribution (log scale) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["yield_kg_ha"], bins=80, color=PALETTE, alpha=0.8, edgecolor="none")
    ax.set_xlabel("Yield (kg/ha)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Global yield distribution — all crops (1990–2023)", fontsize=13)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/01_yield_distribution.png", dpi=150)
    plt.close()
    print("  Saved 01_yield_distribution.png")

    # ── 2. Yield trend over time (top 6 crops, global mean) ─────────────────
    top6 = df["Item"].value_counts().head(6).index.tolist()
    trend = (df[df["Item"].isin(top6)]
               .groupby(["Year","Item"])["yield_kg_ha"]
               .mean().reset_index())
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#3B6D11","#639922","#97C459","#185FA5","#378ADD","#D85A30"]
    for i, crop in enumerate(top6):
        sub = trend[trend["Item"]==crop]
        ax.plot(sub["Year"], sub["yield_kg_ha"]/1000,
                label=crop, linewidth=2, color=colors[i])
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Avg yield (tonnes/ha)", fontsize=12)
    ax.set_title("Yield trends for top 6 crops — global average", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/02_yield_trends.png", dpi=150)
    plt.close()
    print("  Saved 02_yield_trends.png")

    # ── 3. Boxplot: yield by crop ────────────────────────────────────────────
    top10 = df["Item"].value_counts().head(10).index.tolist()
    sub = df[df["Item"].isin(top10)].copy()
    sub["yield_t_ha"] = sub["yield_kg_ha"] / 1000
    fig, ax = plt.subplots(figsize=(13, 6))
    order = sub.groupby("Item")["yield_t_ha"].median().sort_values(ascending=False).index
    sns.boxplot(data=sub, x="Item", y="yield_t_ha", order=order,
                color="#97C459", linewidth=0.8, fliersize=2, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Yield (t/ha)", fontsize=12)
    ax.set_title("Yield distribution by crop — 1990–2023", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/03_yield_by_crop.png", dpi=150)
    plt.close()
    print("  Saved 03_yield_by_crop.png")

    # ── 4. Correlation heatmap ───────────────────────────────────────────────
    num_cols = ["yield_kg_ha","yield_lag1","yield_lag3","yield_roll5",
                "area_harvested_ha","production_tonnes","year_offset","crop_code","area_code"]
    corr = df[num_cols].corr()
    labels = ["Yield","Yield lag1","Yield lag3","Yield roll5",
              "Area (ha)","Production (t)","Year offset","Crop code","Area code"]
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="YlGn", linewidths=0.4,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size":8})
    ax.set_title("Feature correlation matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close()
    print("  Saved 04_correlation_heatmap.png")

    # ── 5. Top 15 producing countries (latest year) ──────────────────────────
    latest = df[df["Year"] == df["Year"].max()]
    top15c = (latest.groupby("Area")["production_tonnes"]
                    .sum().nlargest(15).reset_index())
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top15c["Area"][::-1], top15c["production_tonnes"][::-1]/1e6,
                   color=PALETTE, alpha=0.85)
    ax.set_xlabel("Total production (million tonnes)", fontsize=11)
    ax.set_title(f"Top 15 countries by total crop production ({df['Year'].max()})", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/05_top_countries.png", dpi=150)
    plt.close()
    print("  Saved 05_top_countries.png")

    print("EDA complete — all plots saved to reports/")
    return df


if __name__ == "__main__":
    run_eda()
