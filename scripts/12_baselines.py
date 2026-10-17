# 12_baselines.py - seasonal baselines, Random Forest and TimeSeriesSplit CV on real IMD rainfall
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import rain_common as rc

df = rc.load_monthly()
d = rc.make_features(df)
tr, va, te = rc.split(d)
print(f"rows  train={len(tr)}  val={len(va)}  test={len(te)}")

rows = []
def add(name, split_name, y, p):
    rmse, mae, r2 = rc.metrics(y, p)
    rows.append(dict(model=name, split=split_name, RMSE=round(rmse, 2), MAE=round(mae, 2), R2=round(r2, 4)))

for nm, part in (("val", va), ("test", te)):
    add("Seasonal naive (same month last year)", nm, part.y, part.seasonal_naive)
    add("Climatology (train-year monthly mean)", nm, part.y, part.clim_next)

rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(tr[rc.FEATS], tr.y)
for nm, part in (("val", va), ("test", te)):
    add("Random Forest", nm, part.y, rf.predict(part[rc.FEATS]))

# TimeSeriesSplit on train+val, ordered by target date (all cities together)
dev = pd.concat([tr, va]).sort_values("target_date").reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=5)
cv = []
for k, (a, b) in enumerate(tscv.split(dev), 1):
    m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
    m.fit(dev.loc[a, rc.FEATS], dev.loc[a, "y"])
    rmse, mae, r2 = rc.metrics(dev.loc[b, "y"], m.predict(dev.loc[b, rc.FEATS]))
    sn = rc.metrics(dev.loc[b, "y"], dev.loc[b, "seasonal_naive"])
    cv.append(dict(fold=k, RF_RMSE=round(rmse, 2), RF_R2=round(r2, 4), naive_RMSE=round(sn[0], 2), naive_R2=round(sn[2], 4)))
cv = pd.DataFrame(cv)

res = pd.DataFrame(rows)
print("\n=== Held-out results ===\n", res.to_string(index=False))
print("\n=== TimeSeriesSplit (5 folds, expanding window) ===\n", cv.to_string(index=False))
print("\nCV mean RF R2 = %.4f  | seasonal-naive mean R2 = %.4f" % (cv.RF_R2.mean(), cv.naive_R2.mean()))
imp = pd.Series(rf.feature_importances_, index=rc.FEATS).sort_values(ascending=False).round(3)
print("\nRF feature importance:\n", imp.to_string())

res.to_csv(rc.OUT_DIR + "/baselines_results.csv", index=False)
cv.to_csv(rc.OUT_DIR + "/timeseries_cv.csv", index=False)
print("\nSaved CSVs to", rc.OUT_DIR)
