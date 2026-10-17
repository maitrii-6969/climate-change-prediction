# 13_lightgbm_rainfall.py - LightGBM and XGBoost for next-month rainfall (real IMD data)
import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
import rain_common as rc

d = rc.make_features(rc.load_monthly())
tr, va, te = rc.split(d)
X = rc.FEATS

lg = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
lg.fit(tr[X], tr.y, eval_set=[(va[X], va.y)], callbacks=[lgb.early_stopping(50, verbose=False)])
xg = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                      early_stopping_rounds=50, random_state=42)
xg.fit(tr[X], tr.y, eval_set=[(va[X], va.y)], verbose=False)
print("best iterations: LightGBM", lg.best_iteration_, "| XGBoost", xg.best_iteration)

rows = []
for name, m in (("LightGBM", lg), ("XGBoost", xg)):
    for nm, part in (("val", va), ("test", te)):
        rmse, mae, r2 = rc.metrics(part.y, m.predict(part[X]))
        rows.append(dict(model=name, split=nm, RMSE=round(rmse, 2), MAE=round(mae, 2), R2=round(r2, 4)))
res = pd.DataFrame(rows)
print(res.to_string(index=False))

# per-city test RMSE for LightGBM
te = te.assign(pred=lg.predict(te[X]))
pc = te.groupby("city").apply(lambda g: pd.Series(dict(RMSE=rc.metrics(g.y, g.pred)[0], MAE=rc.metrics(g.y, g.pred)[1]))).round(1)
print("\nLightGBM test error by city:\n", pc.to_string())

res.to_csv(rc.OUT_DIR + "/lightgbm_xgboost_results.csv", index=False)
pc.to_csv(rc.OUT_DIR + "/lightgbm_by_city.csv")
lg.booster_.save_model(rc.OUT_DIR + "/lightgbm_rainfall.txt")
