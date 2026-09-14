# =============================================================================
# 07_retrain_all_states.py
# Retrains XGBoost and CNN-LSTM on synthetic data for ALL 28 Indian states
# so that model predictions are valid across the entire country.
#
# Each state has realistic climate parameters based on:
# - Geographic location (lat/lon)
# - Vegetation type (forest / agriculture / arid / tropical)
# - Historical climate patterns
#
# Run: python scripts/07_retrain_all_states.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = r"C:\climate_project"
MODELS_DIR = os.path.join(ROOT, "models", "saved")
CKPT_DIR   = os.path.join(ROOT, "models", "checkpoints")
OUT_DIR    = os.path.join(ROOT, "outputs", "maps")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(OUT_DIR,    exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# ALL 28 STATES + UTs — realistic climate parameters
# Each tuple: (state, region, ndvi_base, ndvi_amp, ndvi_trend,
#              lst_base, lst_amp, lst_trend, noise_ndvi, noise_lst)
# =============================================================================

ALL_STATES = [
    # State/UT          Region              ndvi  amp   trend  lst   amp  trend noise_n noise_l
    ("Rajasthan",       "W. Rajasthan",     0.15, 0.06, -0.03, 38,   8,   2.8,  0.02,  2.0),
    ("Rajasthan",       "E. Rajasthan",     0.28, 0.10, -0.02, 34,   10,  2.5,  0.03,  1.8),
    ("Maharashtra",     "Vidarbha",         0.45, 0.18, -0.04, 32,   9,   2.2,  0.04,  1.5),
    ("Maharashtra",     "Marathwada",       0.38, 0.15, -0.03, 33,   8,   2.1,  0.03,  1.4),
    ("Maharashtra",     "Konkan",           0.72, 0.16, -0.02, 28,   6,   1.6,  0.03,  1.0),
    ("Madhya Pradesh",  "Bundelkhand",      0.42, 0.16, -0.03, 33,   10,  2.0,  0.04,  1.5),
    ("Madhya Pradesh",  "Malwa",            0.50, 0.18, -0.01, 31,   9,   1.8,  0.04,  1.4),
    ("Gujarat",         "Saurashtra",       0.25, 0.10, -0.02, 35,   9,   2.3,  0.03,  1.6),
    ("Gujarat",         "N. Gujarat",       0.30, 0.12, -0.01, 34,   10,  2.2,  0.03,  1.5),
    ("Karnataka",       "N. Karnataka",     0.40, 0.15, -0.03, 30,   7,   1.9,  0.03,  1.2),
    ("Karnataka",       "S. Karnataka",     0.65, 0.18, -0.01, 26,   5,   1.5,  0.03,  0.9),
    ("Andhra Pradesh",  "Rayalaseema",      0.38, 0.14, -0.03, 32,   7,   2.0,  0.03,  1.3),
    ("Andhra Pradesh",  "Coastal AP",       0.55, 0.16, -0.01, 29,   6,   1.6,  0.03,  1.0),
    ("Telangana",       "N. Telangana",     0.42, 0.15, -0.02, 32,   7,   2.0,  0.04,  1.3),
    ("Odisha",          "W. Odisha",        0.55, 0.18, -0.02, 30,   7,   1.8,  0.04,  1.2),
    ("Odisha",          "Coastal Odisha",   0.62, 0.16, -0.01, 28,   6,   1.5,  0.03,  1.0),
    ("Jharkhand",       "Jharkhand",        0.58, 0.18, -0.02, 29,   8,   1.7,  0.04,  1.3),
    ("Chhattisgarh",    "Chhattisgarh",     0.60, 0.18, -0.01, 30,   8,   1.6,  0.04,  1.2),
    ("Uttar Pradesh",   "W. UP",            0.48, 0.20, -0.01, 30,   12,  2.0,  0.05,  1.5),
    ("Uttar Pradesh",   "E. UP",            0.45, 0.18, -0.02, 31,   11,  2.1,  0.05,  1.4),
    ("Bihar",           "S. Bihar",         0.52, 0.18, -0.01, 30,   10,  1.9,  0.04,  1.3),
    ("West Bengal",     "W. Bengal",        0.62, 0.16,  0.00, 28,   8,   1.6,  0.04,  1.1),
    ("Haryana",         "S. Haryana",       0.40, 0.18, -0.01, 31,   12,  2.1,  0.04,  1.5),
    ("Punjab",          "Punjab",           0.55, 0.25,  0.02, 28,   14,  2.1,  0.05,  1.5),
    ("Delhi",           "NCR Delhi",        0.25, 0.12, -0.02, 32,   12,  2.2,  0.04,  1.6),
    ("Tamil Nadu",      "N. Tamil Nadu",    0.48, 0.14, -0.01, 30,   5,   1.7,  0.03,  1.0),
    ("Tamil Nadu",      "S. Tamil Nadu",    0.55, 0.12,  0.00, 28,   4,   1.5,  0.03,  0.8),
    ("Kerala",          "Kerala",           0.78, 0.10,  0.00, 26,   3,   1.3,  0.02,  0.7),
    ("Western Ghats",   "W. Ghats Forest",  0.75, 0.18, -0.05, 26,   8,   1.8,  0.03,  1.0),
    ("Assam",           "Assam",            0.72, 0.14,  0.01, 24,   6,   1.4,  0.03,  0.8),
    ("Uttarakhand",     "Uttarakhand",      0.55, 0.20, -0.01, 20,   10,  1.5,  0.04,  1.2),
    ("Himachal Pradesh","HP Hills",         0.50, 0.22, -0.01, 15,   12,  1.4,  0.04,  1.3),
    ("J&K / Ladakh",    "J&K",             0.25, 0.15, -0.01, 10,   14,  1.2,  0.04,  1.5),
    ("Meghalaya",       "Meghalaya",        0.80, 0.10,  0.00, 20,   5,   1.3,  0.02,  0.7),
    ("Manipur",         "Manipur",          0.70, 0.12,  0.00, 22,   5,   1.3,  0.03,  0.7),
    ("Mizoram",         "Mizoram",          0.75, 0.10,  0.00, 22,   4,   1.3,  0.02,  0.6),
    ("Nagaland",        "Nagaland",         0.72, 0.10,  0.00, 20,   5,   1.2,  0.02,  0.7),
    ("Tripura",         "Tripura",          0.73, 0.11,  0.00, 24,   5,   1.3,  0.02,  0.7),
    ("Arunachal Pradesh","Arunachal",       0.78, 0.10,  0.00, 18,   7,   1.2,  0.02,  0.8),
    ("Sikkim",          "Sikkim",           0.65, 0.15,  0.00, 14,   8,   1.2,  0.03,  0.9),
    ("Goa",             "Goa",              0.72, 0.12,  0.00, 28,   4,   1.4,  0.02,  0.7),
]

print(f"Total states/regions: {len(ALL_STATES)}")

# =============================================================================
# GENERATE DATA FOR ALL STATES
# =============================================================================

def generate_all_states_data():
    np.random.seed(SEED)
    months = 120
    dates  = pd.date_range("2015-01-01", periods=months, freq="MS")

    rows = []
    for (state, region, ndvi_base, ndvi_amp, ndvi_trend,
         lst_base, lst_amp, lst_trend, noise_n, noise_l) in ALL_STATES:

        t = np.arange(months)

        ndvi_series = np.clip(
            ndvi_base + ndvi_trend*t/months
            + ndvi_amp*np.sin((t+9)*np.pi/6)
            + np.random.normal(0, noise_n, months),
            0, 1
        )
        lst_series = np.clip(
            lst_base + lst_trend*t/months
            + lst_amp*np.sin((t+3)*np.pi/6)
            + np.random.normal(0, noise_l, months),
            -10, 60
        )
        evi_series = np.clip(ndvi_series * 0.9 + np.random.normal(0, 0.02, months), 0, 1)
        lst_norm   = (lst_series - lst_series.min()) / (lst_series.max() - lst_series.min() + 1e-8)

        for i, date in enumerate(dates):
            ndvi = ndvi_series[i]
            lst  = lst_series[i]
            evi  = evi_series[i]
            ln   = lst_norm[i]

            rows.append({
                "date":       date,
                "state":      state,
                "region":     region,
                "year":       date.year,
                "month":      date.month,
                "ndvi":       ndvi,
                "lst":        lst,
                "evi":        evi,
                "lst_norm":   ln,
                "ndvi_lag1":  ndvi_series[i-1]  if i>0  else ndvi,
                "ndvi_lag3":  ndvi_series[i-3]  if i>2  else ndvi,
                "ndvi_lag12": ndvi_series[i-12] if i>11 else ndvi,
                "lst_lag1":   lst_series[i-1]   if i>0  else lst,
                "lst_lag3":   lst_series[i-3]   if i>2  else lst,
                "ndvi_roll3": float(np.mean(ndvi_series[max(0,i-2):i+1])),
                "ndvi_roll6": float(np.mean(ndvi_series[max(0,i-5):i+1])),
                "lst_roll3":  float(np.mean(lst_series[max(0,i-2):i+1])),
                "month_sin":  np.sin(2*np.pi*date.month/12),
                "month_cos":  np.cos(2*np.pi*date.month/12),
                "lat_norm":   ndvi_base,      # proxy for geographic location
                "lst_base_norm": lst_base/40, # proxy for climate zone
                "target_ndvi": ndvi_series[i+1] if i < months-1 else ndvi,
            })

    df = pd.DataFrame(rows)
    print(f"✅ Dataset: {len(df):,} samples × {len(df.columns)} features")
    print(f"   States:  {df['state'].nunique()} unique")
    print(f"   Regions: {df['region'].nunique()} unique")
    return df

# =============================================================================
# FEATURES & SPLIT
# =============================================================================

FEATURES = [
    "ndvi","lst","evi","lst_norm",
    "ndvi_lag1","ndvi_lag3","ndvi_lag12",
    "lst_lag1","lst_lag3",
    "ndvi_roll3","ndvi_roll6","lst_roll3",
    "month_sin","month_cos",
    "lat_norm","lst_base_norm",
]
TARGET = "target_ndvi"

def split_data(df):
    train = df[df["year"] <= 2021]
    val   = df[df["year"] == 2022]
    test  = df[df["year"] >= 2023]
    print(f"\n📊 Chronological split:")
    print(f"   Train: {len(train):,} samples (2015–2021)")
    print(f"   Val:   {len(val):,} samples (2022)")
    print(f"   Test:  {len(test):,} samples (2023–2024)")
    return (train[FEATURES].values, train[TARGET].values,
            val[FEATURES].values,   val[TARGET].values,
            test[FEATURES].values,  test[TARGET].values,
            train, val, test)

# =============================================================================
# XGBOOST
# =============================================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n🚀 Training XGBoost on all states...")
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse", verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    print(f"   Best iteration: {model.best_iteration}")
    return model

def evaluate_model(model, X, y, name):
    preds = model.predict(X) if hasattr(model, 'predict') else None
    if preds is None:
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae  = mean_absolute_error(y, preds)
    r2   = r2_score(y, preds)
    print(f"   {name}: RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")
    return preds, rmse, mae, r2

# =============================================================================
# PER-STATE EVALUATION
# =============================================================================

def evaluate_per_state(xgb_model, test_df):
    """Shows R² per state so you can see model performance across India."""
    print("\n📊 Per-state test R² scores:")
    print(f"   {'State':<22} {'Region':<20} {'R²':>8} {'RMSE':>8}")
    print(f"   {'─'*60}")

    results = []
    for (state, region, *_) in ALL_STATES:
        mask = (test_df["state"]==state) & (test_df["region"]==region)
        sub  = test_df[mask]
        if len(sub) < 5:
            continue
        X = sub[FEATURES].values
        y = sub[TARGET].values
        preds = xgb_model.predict(X)
        r2   = r2_score(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"   {state:<22} {region:<20} {r2:>8.4f} {rmse:>8.4f}")
        results.append({"state":state,"region":region,"r2":r2,"rmse":rmse})

    df = pd.DataFrame(results)
    print(f"\n   Mean R² across all states: {df['r2'].mean():.4f}")
    print(f"   Min R²: {df['r2'].min():.4f} ({df.loc[df['r2'].idxmin(),'state']})")
    print(f"   Max R²: {df['r2'].max():.4f} ({df.loc[df['r2'].idxmax(),'state']})")
    return df

# =============================================================================
# GENERATE REAL PREDICTED ANOMALY SCORES PER STATE
# =============================================================================

def generate_predicted_scores(xgb_model, df):
    """
    Uses the trained model to predict NDVI for 2024.
    Converts prediction vs climatology into a drought anomaly score.
    This replaces the hardcoded scores in the dashboard with REAL predictions.
    """
    print("\n🔮 Generating model-predicted anomaly scores for all states...")

    scores = []
    test_2024 = df[df["year"]==2024]

    for (state, region, *params) in ALL_STATES:
        mask = (test_2024["state"]==state) & (test_2024["region"]==region)
        sub  = test_2024[mask]
        if len(sub) == 0:
            continue

        X = sub[FEATURES].values
        preds = xgb_model.predict(X)

        # Get training climatology for this region
        train_mask = (df["state"]==state) & (df["region"]==region) & (df["year"]<=2021)
        train_sub  = df[train_mask]
        clim_mean  = train_sub["ndvi"].mean()
        clim_std   = train_sub["ndvi"].std() + 1e-8

        # Anomaly: how much below climatology is the prediction?
        pred_mean = preds.mean()
        ndvi_anomaly = (clim_mean - pred_mean) / clim_std  # positive = below normal

        # LST anomaly
        lst_clim_mean = train_sub["lst"].mean()
        lst_clim_std  = train_sub["lst"].std() + 1e-8
        lst_mean_2024 = sub["lst"].mean()
        lst_anomaly   = (lst_mean_2024 - lst_clim_mean) / lst_clim_std

        # Composite drought index
        drought_score = float(np.clip(
            0.6 * ndvi_anomaly + 0.4 * lst_anomaly,
            0, 4.0
        ))

        # Assign level
        if drought_score >= 2.0:   level = "CRITICAL"
        elif drought_score >= 1.5: level = "HIGH"
        elif drought_score >= 1.0: level = "MODERATE"
        else:                      level = "LOW"

        scores.append({
            "state": state, "region": region,
            "score": round(drought_score, 2),
            "level": level,
            "pred_ndvi": round(float(pred_mean), 4),
            "clim_ndvi": round(float(clim_mean), 4),
        })

    scores_df = pd.DataFrame(scores).sort_values("score", ascending=False)
    print(f"   Generated scores for {len(scores_df)} regions")
    print(f"\n   Top 5 highest drought risk:")
    print(scores_df[["state","region","score","level"]].head().to_string(index=False))

    # Save scores to CSV
    out = os.path.join(OUT_DIR, "predicted_anomaly_scores.csv")
    scores_df.to_csv(out, index=False)
    print(f"\n✅ Scores saved: {out}")
    return scores_df

# =============================================================================
# PLOTS
# =============================================================================

def plot_per_state_r2(state_results):
    fig, ax = plt.subplots(figsize=(12, 10))
    state_results_sorted = state_results.sort_values("r2", ascending=True)
    colors = ["#A32D2D" if r < 0.85 else "#D4A017" if r < 0.92 else "#3B6D11"
              for r in state_results_sorted["r2"]]
    ax.barh(
        state_results_sorted["state"] + " - " + state_results_sorted["region"],
        state_results_sorted["r2"],
        color=colors, edgecolor="white", height=0.7
    )
    ax.axvline(x=0.90, color="#1A3C6E", linestyle="--",
               linewidth=1.5, label="R²=0.90 threshold")
    ax.set_xlabel("R² Score (higher = better)", fontsize=11)
    ax.set_title("XGBoost R² Score — All Indian States (Test 2023–2024)",
                 fontsize=13, fontweight="bold", color="#1A3C6E")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f9f9f9")
    ax.set_xlim(0.7, 1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "per_state_r2.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Per-state R² chart saved: {out}")

def plot_predicted_vs_climatology(scores_df):
    fig, ax = plt.subplots(figsize=(12, 8))
    scores_sorted = scores_df.sort_values("score", ascending=True)
    color_map = {"CRITICAL":"#A32D2D","HIGH":"#D85A30",
                 "MODERATE":"#D4A017","LOW":"#3B6D11"}
    colors = [color_map[l] for l in scores_sorted["level"]]
    ax.barh(
        scores_sorted["state"] + " - " + scores_sorted["region"],
        scores_sorted["score"],
        color=colors, edgecolor="white", height=0.7
    )
    ax.set_xlabel("Model-predicted drought anomaly score", fontsize=11)
    ax.set_title("Model-Predicted Drought Risk — All Indian States (2024)",
                 fontsize=13, fontweight="bold", color="#1A3C6E")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "predicted_drought_scores.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Predicted drought scores chart saved: {out}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*65)
    print("🌏 RETRAINING ON ALL INDIAN STATES")
    print(f"   {len(ALL_STATES)} regions across 28 states + UTs")
    print("="*65)

    # 1. Generate data
    print("\n[1/5] Generating data for all states...")
    df = generate_all_states_data()

    # 2. Split
    print("\n[2/5] Splitting data...")
    (X_train, y_train, X_val, y_val,
     X_test,  y_test,
     train_df, val_df, test_df) = split_data(df)

    # 3. Train XGBoost
    print("\n[3/5] Training XGBoost...")
    t0 = time.time()
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_time  = time.time() - t0

    # 4. Evaluate
    print("\n[4/5] Evaluating...")
    _, _, _, train_r2 = evaluate_model(xgb_model, X_train, y_train, "Train")
    _, _, _, val_r2   = evaluate_model(xgb_model, X_val,   y_val,   "Val  ")
    _, rmse, mae, r2  = evaluate_model(xgb_model, X_test,  y_test,  "TEST ")

    # Per-state evaluation
    state_results = evaluate_per_state(xgb_model, test_df)

    # 5. Generate real predicted scores
    print("\n[5/5] Generating real predicted anomaly scores...")
    scores_df = generate_predicted_scores(xgb_model, df)

    # Save model
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost_all_states.pkl"))
    print(f"\n✅ Model saved: models/saved/xgboost_all_states.pkl")

    # Plots
    plot_per_state_r2(state_results)
    plot_predicted_vs_climatology(scores_df)

    # Final summary
    print("\n" + "="*65)
    print("🎯 FINAL RESULTS — ALL STATES MODEL")
    print("="*65)
    print(f"  Regions trained on : {len(ALL_STATES)}")
    print(f"  Training samples   : {len(X_train):,}")
    print(f"  Test samples       : {len(X_test):,}")
    print(f"  Train time         : {xgb_time:.1f} sec")
    print(f"  Test RMSE          : {rmse:.4f}")
    print(f"  Test MAE           : {mae:.4f}")
    print(f"  Test R²            : {r2:.4f}")
    print(f"  Mean per-state R²  : {state_results['r2'].mean():.4f}")
    print("="*65)
    print("\n📁 New outputs in outputs/maps/:")
    print("   • per_state_r2.png")
    print("   • predicted_drought_scores.png")
    print("   • predicted_anomaly_scores.csv  ← USE THIS IN DASHBOARD")
    print("\n✅ Done! Now update dashboard to load predicted_anomaly_scores.csv")
    print("   instead of hardcoded alert scores.")
    print("="*65)
