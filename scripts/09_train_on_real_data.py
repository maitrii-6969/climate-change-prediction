# 09_train_on_real_data.py
# Trains XGBoost on REAL ERA5 data
# Run: python scripts/09_train_on_real_data.py

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

ROOT      = r"C:\climate_project"
IDX_DIR   = os.path.join(ROOT, "data", "indices")
MODELS_DIR= os.path.join(ROOT, "models", "saved")
OUT_DIR   = os.path.join(ROOT, "outputs", "maps")
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

CITIES = {
    "Mumbai":    {"lat": 19.1, "lon": 72.9},
    "Delhi":     {"lat": 28.6, "lon": 77.2},
    "Chennai":   {"lat": 13.1, "lon": 80.3},
    "Kolkata":   {"lat": 22.6, "lon": 88.4},
    "Bangalore": {"lat": 12.9, "lon": 77.6},
    "Hyderabad": {"lat": 17.4, "lon": 78.5},
    "Jaipur":    {"lat": 26.9, "lon": 75.8},
    "Ahmedabad": {"lat": 23.0, "lon": 72.6},
    "Bhopal":    {"lat": 23.3, "lon": 77.4},
    "Nagpur":    {"lat": 21.1, "lon": 79.1},
}

print("="*60)
print("TRAINING XGBOOST ON REAL ERA5 DATA")
print("="*60)

# Load real data
print("\n[1/5] Loading real ERA5 data...")
lst_ds    = xr.open_dataset(os.path.join(IDX_DIR, "era5_lst_india.nc"))
t2m_ds    = xr.open_dataset(os.path.join(IDX_DIR, "era5_t2m_india.nc"))
precip_ds = xr.open_dataset(os.path.join(IDX_DIR, "era5_precip_india.nc"))

lst    = lst_ds["lst"]
t2m    = t2m_ds["t2m"]
precip = precip_ds["precip"]

if "valid_time" in lst.coords:
    lst    = lst.rename({"valid_time": "time"})
    t2m    = t2m.rename({"valid_time": "time"})
    precip = precip.rename({"valid_time": "time"})

times = pd.DatetimeIndex(lst.time.values)
print(f"   Time range: {times[0].date()} to {times[-1].date()}")
print(f"   Months: {len(times)}")
print(f"   Cities: {len(CITIES)}")

# Build dataset
print("\n[2/5] Building feature dataset from real data...")
rows = []
for city, coords in CITIES.items():
    lst_ts    = lst.sel(latitude=coords["lat"], longitude=coords["lon"], method="nearest").values
    t2m_ts    = t2m.sel(latitude=coords["lat"], longitude=coords["lon"], method="nearest").values
    precip_ts = precip.sel(latitude=coords["lat"], longitude=coords["lon"], method="nearest").values

    # Normalise LST to 0-1
    lst_norm = (lst_ts - lst_ts.min()) / (lst_ts.max() - lst_ts.min() + 1e-8)

    for i, date in enumerate(times):
        rows.append({
            "date":       date,
            "city":       city,
            "year":       date.year,
            "month":      date.month,
            "lst":        lst_ts[i],
            "t2m":        t2m_ts[i],
            "precip":     precip_ts[i],
            "lst_norm":   lst_norm[i],
            "lst_lag1":   lst_ts[i-1]    if i > 0  else lst_ts[i],
            "lst_lag3":   lst_ts[i-3]    if i > 2  else lst_ts[i],
            "lst_lag12":  lst_ts[i-12]   if i > 11 else lst_ts[i],
            "t2m_lag1":   t2m_ts[i-1]    if i > 0  else t2m_ts[i],
            "precip_lag1":precip_ts[i-1] if i > 0  else precip_ts[i],
            "precip_lag3":precip_ts[i-3] if i > 2  else precip_ts[i],
            "lst_roll3":  float(np.mean(lst_ts[max(0,i-2):i+1])),
            "lst_roll6":  float(np.mean(lst_ts[max(0,i-5):i+1])),
            "t2m_roll3":  float(np.mean(t2m_ts[max(0,i-2):i+1])),
            "month_sin":  np.sin(2*np.pi*date.month/12),
            "month_cos":  np.cos(2*np.pi*date.month/12),
            "target_lst": lst_ts[i+1] if i < len(times)-1 else lst_ts[i],
        })

df = pd.DataFrame(rows)
print(f"   Total samples: {len(df):,}")

FEATURES = [
    "lst","t2m","precip","lst_norm",
    "lst_lag1","lst_lag3","lst_lag12",
    "t2m_lag1","precip_lag1","precip_lag3",
    "lst_roll3","lst_roll6","t2m_roll3",
    "month_sin","month_cos",
]
TARGET = "target_lst"

# Chronological split
print("\n[3/5] Splitting data...")
train = df[df["year"] <= 2021]
val   = df[df["year"] == 2022]
test  = df[df["year"] >= 2023]

X_train = train[FEATURES].values; y_train = train[TARGET].values
X_val   = val[FEATURES].values;   y_val   = val[TARGET].values
X_test  = test[FEATURES].values;  y_test  = test[TARGET].values

print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# Train
print("\n[4/5] Training XGBoost on REAL data...")
model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1,
    reg_lambda=1.0, random_state=SEED,
    n_jobs=-1, early_stopping_rounds=30,
    eval_metric="rmse", verbosity=0,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Evaluate
def evaluate(X, y, name):
    preds = model.predict(X)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    print(f"   {name}: RMSE={rmse:.4f}°C | MAE={mae:.4f} | R²={r2:.4f}")
    return preds, rmse, mae, r2

print("\n[5/5] Evaluating...")
evaluate(X_train, y_train, "Train     ")
evaluate(X_val,   y_val,   "Validation")
test_preds, rmse, mae, r2 = evaluate(X_test, y_test, "TEST      ")

# Save model
joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_real_era5.pkl"))
print(f"\n✅ Model saved: models/saved/xgboost_real_era5.pkl")

# Plot predictions vs actual
print("\nGenerating plots...")
city = "Mumbai"
city_test = test[test["city"]==city].copy()
city_preds = model.predict(city_test[FEATURES].values)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(city_test["date"].values, city_test[TARGET].values,
        color="#2e8b57", linewidth=2, label="Actual LST (REAL ERA5)", alpha=0.9)
ax.plot(city_test["date"].values, city_preds,
        color="#D85A30", linewidth=2, linestyle="--",
        label="XGBoost prediction", alpha=0.85)
ax.set_title(f"REAL ERA5 Data — XGBoost LST Prediction vs Actual ({city})\nTest period: 2023–2024",
             fontsize=13, fontweight="bold", color="#1A3C6E")
ax.set_xlabel("Date"); ax.set_ylabel("LST (°C)")
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_facecolor("#f9f9f9")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "real_data_predictions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ Prediction plot saved: real_data_predictions.png")

# Feature importance
fi = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
fi = fi.sort_values("importance", ascending=True)
fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#2E75B6" if i >= len(fi)-5 else "#A9C4E4" for i in range(len(fi))]
ax.barh(fi["feature"], fi["importance"], color=colors, edgecolor="white", height=0.7)
ax.set_title("Real ERA5 — XGBoost Feature Importance (LST Prediction)",
             fontsize=13, fontweight="bold", color="#1A3C6E")
ax.set_xlabel("Importance score")
ax.grid(True, axis="x", linestyle="--", alpha=0.4)
ax.set_facecolor("#f9f9f9")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "real_data_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ Feature importance saved: real_data_feature_importance.png")

print("\n" + "="*60)
print("FINAL RESULTS — TRAINED ON REAL ERA5 DATA")
print("="*60)
print(f"  Data source  : REAL ERA5 (not synthetic!)")
print(f"  Cities       : {len(CITIES)}")
print(f"  Time period  : 2014–2024 (120 months)")
print(f"  Test RMSE    : {rmse:.4f} °C")
print(f"  Test MAE     : {mae:.4f}")
print(f"  Test R²      : {r2:.4f}")
print(f"  Best iter    : {model.best_iteration}")
print("="*60)
print("\n✅ Done! Your models now train on REAL satellite data!")

lst_ds.close(); t2m_ds.close(); precip_ds.close()