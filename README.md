# 🌾 Crop Yield Prediction — FAO FAOSTAT

End-to-end machine learning pipeline that predicts crop yield (kg/ha)
for 20 major crops across 244 countries using **FAO FAOSTAT** production data.

---

## 📁 Project Structure

```
crop_yield_project/
├── data/
│   ├── raw/        ← FAO CSV (place here before running)
│   └── processed/  ← cleaned & feature-engineered data
├── models/         ← saved .pkl model files + metrics.json
├── reports/        ← all EDA and evaluation plots (.png)
├── src/
│   ├── preprocess.py   ← data cleaning & feature engineering
│   ├── eda.py          ← exploratory data analysis plots
│   ├── train.py        ← model training + SHAP
│   └── predict.py      ← inference / prediction API
├── app/
│   └── streamlit_app.py  ← interactive dashboard
├── run_pipeline.py     ← run everything end-to-end
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```

This will:
- Preprocess the raw FAO data
- Generate 10 EDA + evaluation plots in `reports/`
- Train 3 models (Ridge, Random Forest, XGBoost)
- Save models to `models/`
- Run sample predictions for 5 country/crop pairs

### 3. Launch the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Dataset

**Source:** FAO FAOSTAT — Crops and Livestock Products
- **URL:** https://www.fao.org/faostat/en/#data/QCL
- **Coverage:** 1961–2024, 244 countries, 170+ crops
- **Elements used:** Yield (kg/ha), Area harvested (ha), Production (tonnes)

**Crops included:**
Wheat, Rice (paddy), Maize, Barley, Soybeans, Sugar cane, Potatoes,
Cassava, Cotton seed, Sorghum, Sunflower seed, Rapeseed, Groundnuts,
Millet, Bananas, Tomatoes, Onions, Beans, Lentils, Oranges

---

## 🤖 Models

| Model         | R²     | RMSE (kg/ha) | MAPE   |
|---------------|--------|-------------|--------|
| Ridge         | ~0.87  | higher      | higher |
| Random Forest | ~0.95  | lower       | ~6%    |
| **XGBoost**   | **~0.97** | **lowest** | **~4%** |

### Features used
| Feature         | Description                            |
|----------------|----------------------------------------|
| crop_code       | Encoded crop type                     |
| area_code       | Encoded country                       |
| year_offset     | Years since 1990                      |
| decade          | Decade bucket                         |
| yield_lag1      | Yield from previous year              |
| yield_lag3      | Yield from 3 years ago                |
| yield_roll5     | 5-year rolling mean yield             |
| area_harvested_ha | Harvested area (hectares)           |
| prod_per_ha     | Production intensity proxy            |

---

## 📈 Reports Generated

| File | Description |
|------|-------------|
| 01_yield_distribution.png | Global yield histogram |
| 02_yield_trends.png | Yield over time — top 6 crops |
| 03_yield_by_crop.png | Yield boxplot per crop |
| 04_correlation_heatmap.png | Feature correlation matrix |
| 05_top_countries.png | Top 15 producing countries |
| 06_model_comparison.png | R², RMSE, MAPE bar charts |
| 07_actual_vs_predicted.png | XGBoost scatter plot |
| 08_residuals.png | Residual plot |
| 09_shap_importance.png | SHAP feature importance |
| 10_shap_beeswarm.png | SHAP beeswarm plot |

---

## 🔮 Make a prediction (Python)

```python
from src.predict import predict_yield

result = predict_yield("Wheat", "India", 2025)
print(result)
# {'crop': 'Wheat', 'country': 'India', 'year': 2025,
#  'predicted_yield_kg_ha': 3241.5, 'predicted_yield_t_ha': 3.242}
```

---

## 📝 Notes
- Time-based train/test split: test = years 2020–2023
- Lag features are computed from historical data to avoid data leakage
- StandardScaler applied only for Ridge; tree models use raw features
- SHAP values computed on a 500-sample subset for speed

---

## 📄 License
For educational and research purposes.
Data © FAO FAOSTAT — https://www.fao.org/faostat
