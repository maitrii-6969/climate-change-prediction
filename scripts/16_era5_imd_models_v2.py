# 15_era5_imd_models.py
# Clean ERA5 (date-aligned) + IMD rainfall -> next-month models.
#   Target A: next-month rainfall (IMD, real gauge-based)   Target B: next-month skin temperature (ERA5 skt, deg C)
# ERA5 is joined to IMD by (city, month) DATE, never by position. Precip fix: tp is metres/day -> x days_in_month x 1000 = mm.
import os, glob
import numpy as np, pandas as pd, xarray as xr
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import rain_common as rc

ERA5_DIR = os.environ.get("ERA5_DIR", r"C:\climate_project\data\raw\era5")

def load_era5():
    # choose files by CONTENT: the "accumulated" file has tp+ssrd, the "instant" file has t2m+skt+u10+v10,
    # and both must cover 120 months (2015-2024). File names are ignored (old downloads share the same names).
    a = u = None
    for f in sorted(glob.glob(os.path.join(ERA5_DIR, "*.nc"))):
        try:
            d = xr.open_dataset(f)
        except Exception:
            continue
        vars_, n = set(d.data_vars), d.sizes.get("valid_time", d.sizes.get("time", 0))
        print(f"  found {os.path.basename(f)}: vars={sorted(vars_)}, months={n}")
        if {"tp", "ssrd"} <= vars_ and n == 120: a = d
        if {"t2m", "skt", "u10", "v10"} <= vars_ and n == 120: u = d
    if a is None or u is None:
        raise SystemExit("Could not find both ERA5 files with 120 months. Keep ONLY Raghav's two new files "
                         "(tp+ssrd file and t2m+skt+u10+v10 file) in " + ERA5_DIR + " and move any old ERA5 files elsewhere.")
    ta = pd.to_datetime(a.valid_time.values).to_period("M").to_timestamp()
    tu = pd.to_datetime(u.valid_time.values).to_period("M").to_timestamp()
    rows = []
    for city, (lat, lon) in rc.CITIES.items():
        pa = a.sel(latitude=lat, longitude=lon, method="nearest")
        pu = u.sel(latitude=lat, longitude=lon, method="nearest")
        da = pd.DataFrame({"city": city, "date": ta,
                           "era5_tp_mm": pa.tp.values * ta.days_in_month.values * 1000,
                           "ssrd": pa.ssrd.values / 1e6})                    # MJ/m2/day
        du = pd.DataFrame({"city": city, "date": tu, "t2m": pu.t2m.values - 273.15,
                           "skt": pu.skt.values - 273.15,
                           "wind": np.hypot(pu.u10.values, pu.v10.values)})
        rows.append(da.merge(du, on=["city", "date"], validate="one_to_one"))
    return pd.concat(rows, ignore_index=True)

era = load_era5()
imd = rc.load_monthly()
m = imd.merge(era, on=["city", "date"], how="inner", validate="one_to_one")
assert len(m) == 1200, f"join lost rows: {len(m)}"
print("joined rows:", len(m), "(10 cities x 120 months, matched on date)")
print("corr(IMD rain, ERA5 tp) overall: %.3f" % m.rain_mm.corr(m.era5_tp_mm))

def build(target):
    out = []
    for city, g in m.groupby("city"):
        g = g.sort_values("date").reset_index(drop=True)
        f = pd.DataFrame({"city": city, "date": g.date})
        for c in ["rain_mm", "era5_tp_mm", "ssrd", "t2m", "skt", "wind"]:
            f[c] = g[c]; f[c + "_l1"] = g[c].shift(1); f[c + "_l12"] = g[c].shift(11)
        f["roll3_rain"] = g.rain_mm.rolling(3).mean(); f["roll3_skt"] = g.skt.rolling(3).mean()
        nm = (g.date.dt.month % 12) + 1
        f["m_sin"] = np.sin(2*np.pi*nm/12); f["m_cos"] = np.cos(2*np.pi*nm/12)
        f["target_date"] = g.date + pd.offsets.MonthBegin(1)
        f["target_month"] = nm
        f["naive"] = g[target].shift(11)
        f["y"] = g[target].shift(-1)
        out.append(f)
    d = pd.concat(out, ignore_index=True)
    d["city_id"] = d.city.astype("category").cat.codes
    d = d.dropna().reset_index(drop=True)
    tr = d[d.target_date.dt.year <= rc.TRAIN_END]
    d["clim"] = d.set_index(["city", "target_month"]).index.map(
        tr.groupby(["city", "target_month"]).y.mean())
    return d

res = []
for target, label in (("rain_mm", "Rainfall (mm)"), ("skt", "Skin temp (C)")):
    d = build(target)
    X = [c for c in d.columns if c not in ("city", "date", "target_date", "target_month", "y", "naive")]
    yr = d.target_date.dt.year
    tr, va, te = d[yr <= rc.TRAIN_END], d[yr == rc.VAL_YEAR], d[yr >= rc.TEST_START]
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(tr[X], tr.y),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1).fit(tr[X], tr.y),
    }
    for nm, part in (("val", va), ("test", te)):
        for name, p in (("Seasonal naive", part.naive), ("Climatology", part.clim)):
            r = rc.metrics(part.y, p); res.append((label, name, nm, *r))
        for name, mdl in models.items():
            r = rc.metrics(part.y, mdl.predict(part[X])); res.append((label, name, nm, *r))

R = pd.DataFrame(res, columns=["target", "model", "split", "RMSE", "MAE", "R2"]).round(3)
print(R[R.split == "test"].to_string(index=False))
R.to_csv(os.path.join(rc.OUT_DIR, "era5_imd_models.csv"), index=False)
m.to_csv(os.path.join(rc.OUT_DIR, "era5_imd_monthly_joined.csv"), index=False)
print("Saved to", rc.OUT_DIR)
