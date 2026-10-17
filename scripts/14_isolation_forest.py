# 14_isolation_forest.py - anomaly detection on monthly rainfall (real IMD data)
# Flags unusual months per city using rainfall anomaly (vs train-year climatology) + seasonality.
import pandas as pd, numpy as np
from sklearn.ensemble import IsolationForest
import rain_common as rc

df = rc.load_monthly()
df["month"] = df.date.dt.month
clim = df[df.date.dt.year <= rc.TRAIN_END].groupby(["city", "month"]).rain_mm.agg(["mean", "std"]).reset_index()
df = df.merge(clim, on=["city", "month"])
df["anom"] = df.rain_mm - df["mean"]
df["z"] = df.anom / (df["std"] + 1e-6)
df["m_sin"] = np.sin(2 * np.pi * df.month / 12); df["m_cos"] = np.cos(2 * np.pi * df.month / 12)
F = ["z", "anom", "m_sin", "m_cos"]

train = df[df.date.dt.year <= rc.TRAIN_END]
iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42).fit(train[F])
df["flag"] = iso.predict(df[F]) == -1
df["score"] = -iso.score_samples(df[F])          # higher = more anomalous
df["kind"] = np.where(df.z > 0, "excess", "deficit")

test = df[df.date.dt.year >= rc.TEST_START]
print("Flagged months (all years): %d of %d (%.1f%%)" % (df.flag.sum(), len(df), 100 * df.flag.mean()))
print("Flagged in 2023-2024 (unseen): %d of %d" % (test.flag.sum(), len(test)))
print("\nTop 15 most anomalous months in 2023-2024:")
print(test.sort_values("score", ascending=False).head(15)[["city", "date", "rain_mm", "anom", "z", "kind"]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
print("\nFlags per city (2015-2024):\n", df.groupby("city").flag.sum().to_string())

df[["city", "date", "rain_mm", "anom", "z", "score", "flag", "kind"]].to_csv(rc.OUT_DIR + "/rainfall_anomalies.csv", index=False, float_format="%.3f")
