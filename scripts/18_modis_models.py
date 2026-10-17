# 18_modis_models.py - next-month NDVI prediction from REAL MODIS + IMD rainfall.
# Run 17_modis_city_extract.py first. Same split as everything else: train 2015-2021, val 2022, test 2023-2024.
import os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb, lightgbm as lgb
import rain_common as rc

mod = pd.read_csv(os.path.join(rc.OUT_DIR, "modis_city_monthly.csv"), parse_dates=["date"])
imd = rc.load_monthly()
m = mod.merge(imd, on=["city", "date"], how="inner", validate="one_to_one")
print("joined city-months:", len(m), "(expected 1200)")

out = []
for city, g in m.groupby("city"):
    g = g.sort_values("date").reset_index(drop=True)
    f = pd.DataFrame({"city": city, "date": g.date})
    for c in ["ndvi", "evi", "lst_c", "rain_mm"]:
        f[c] = g[c]; f[c + "_l1"] = g[c].shift(1); f[c + "_l2"] = g[c].shift(2); f[c + "_l12"] = g[c].shift(11)
    f["rain_roll3"] = g.rain_mm.rolling(3).mean(); f["rain_roll6"] = g.rain_mm.rolling(6).mean()
    f["ndvi_roll3"] = g.ndvi.rolling(3).mean()
    nm = (g.date.dt.month % 12) + 1
    f["m_sin"] = np.sin(2*np.pi*nm/12); f["m_cos"] = np.cos(2*np.pi*nm/12); f["target_month"] = nm
    f["target_date"] = g.date + pd.offsets.MonthBegin(1)
    f["naive"] = g.ndvi.shift(11); f["y"] = g.ndvi.shift(-1)
    out.append(f)
d = pd.concat(out, ignore_index=True)
d["city_id"] = d.city.astype("category").cat.codes
d = d.dropna().reset_index(drop=True)
yr = d.target_date.dt.year
tr = d[yr <= rc.TRAIN_END]
d["clim"] = d.set_index(["city", "target_month"]).index.map(tr.groupby(["city", "target_month"]).y.mean())
tr, va, te = d[yr <= rc.TRAIN_END], d[yr == rc.VAL_YEAR], d[yr >= rc.TEST_START]
print(f"rows train={len(tr)} val={len(va)} test={len(te)}")
X = [c for c in d.columns if c not in ("city", "date", "target_date", "target_month", "y", "naive")]

rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(tr[X], tr.y)
xg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                      early_stopping_rounds=40, random_state=42).fit(tr[X], tr.y, eval_set=[(va[X], va.y)], verbose=False)
lg = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1
                       ).fit(tr[X], tr.y, eval_set=[(va[X], va.y)], callbacks=[lgb.early_stopping(40, verbose=False)])

rows = []
for nm_, part in (("val", va), ("test", te)):
    for name, p in (("Seasonal naive", part.naive), ("Climatology", part.clim), ("Random Forest", rf.predict(part[X])),
                    ("XGBoost", xg.predict(part[X])), ("LightGBM", lg.predict(part[X]))):
        r = rc.metrics(part.y, p); rows.append((name, nm_, round(r[0], 4), round(r[1], 4), round(r[2], 4)))
R = pd.DataFrame(rows, columns=["model", "split", "RMSE", "MAE", "R2"])
print("\nNext-month NDVI (real MODIS):\n", R[R.split == "test"].to_string(index=False))
print("\nValidation:\n", R[R.split == "val"].to_string(index=False))
imp = pd.Series(rf.feature_importances_, index=X).sort_values(ascending=False).head(8).round(3)
print("\nRF top features:\n", imp.to_string())
R.to_csv(os.path.join(rc.OUT_DIR, "modis_ndvi_models.csv"), index=False)
print("\nSaved modis_ndvi_models.csv to", rc.OUT_DIR)
