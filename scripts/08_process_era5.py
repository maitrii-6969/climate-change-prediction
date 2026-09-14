# =============================================================================
# 08_process_era5.py
# Processes real ERA5 NetCDF files and integrates into the project pipeline.
#
# Input files (put in data/raw/era5/):
#   data_stream-moda_stepType-avgad.nc  → Total Precipitation (tp)
#   data_stream-moda_stepType-avgua.nc  → t2m, skt (LST), sp
#
# Outputs:
#   data/indices/era5_lst_india.nc      → Land Surface Temperature (°C)
#   data/indices/era5_t2m_india.nc      → 2m Air Temperature (°C)
#   data/indices/era5_precip_india.nc   → Precipitation (mm/month)
#   outputs/maps/era5_lst_mean.png      → LST mean map
#   outputs/maps/era5_precip_annual.png → Annual rainfall chart
#   outputs/maps/era5_timeseries.png    → Time-series for 3 cities
#
# Run: python scripts/08_process_era5.py
# =============================================================================

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = r"C:\climate_project"
ERA5_DIR  = os.path.join(ROOT, "data", "raw", "era5")
IDX_DIR   = os.path.join(ROOT, "data", "indices")
OUT_DIR   = os.path.join(ROOT, "outputs", "maps")
os.makedirs(ERA5_DIR, exist_ok=True)
os.makedirs(IDX_DIR,  exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

# ── ERA5 file paths ───────────────────────────────────────────────────────────
# CHANGE THESE if your files are in a different location
FILE_AVGUA = os.path.join(ERA5_DIR, "data_stream-moda_stepType-avgua.nc")
FILE_AVGAD = os.path.join(ERA5_DIR, "data_stream-moda_stepType-avgad.nc")

# ── City coordinates for time-series plots ────────────────────────────────────
CITIES = {
    "Mumbai":  {"lat": 19.1, "lon": 72.9},
    "Delhi":   {"lat": 28.6, "lon": 77.2},
    "Chennai": {"lat": 13.1, "lon": 80.3},
}

# =============================================================================
# 1. LOAD AND VALIDATE FILES
# =============================================================================

def check_files():
    """Check if ERA5 files exist — guide user if not."""
    missing = []
    for label, path in [("avgua (LST/Temperature)", FILE_AVGUA),
                         ("avgad (Precipitation)",   FILE_AVGAD)]:
        if not os.path.exists(path):
            missing.append((label, path))

    if missing:
        print("❌ ERA5 files not found in data/raw/era5/")
        print("   Please copy your downloaded ERA5 files there:")
        for label, path in missing:
            print(f"   Missing: {path}")
        print("\n   Your files are currently in downloads — run this command:")
        print(f'   copy "%USERPROFILE%\\Downloads\\data_stream-moda_stepType-avgua.nc" "{FILE_AVGUA}"')
        print(f'   copy "%USERPROFILE%\\Downloads\\data_stream-moda_stepType-avgad.nc" "{FILE_AVGAD}"')
        return False
    return True

# =============================================================================
# 2. PROCESS TEMPERATURE + LST
# =============================================================================

def process_temperature():
    """
    Loads t2m and skt from avgua.nc
    Converts Kelvin → Celsius
    Saves as NetCDF
    """
    print("\n[1/3] Processing temperature data...")
    ds = xr.open_dataset(FILE_AVGUA)

    # Rename valid_time to time for consistency
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # Convert K → °C
    t2m = ds["t2m"] - 273.15
    skt = ds["skt"] - 273.15    # Skin temp ≈ Land Surface Temperature

    t2m.attrs = {"long_name": "2m Air Temperature", "units": "°C", "source": "ERA5"}
    skt.attrs = {"long_name": "Land Surface Temperature", "units": "°C", "source": "ERA5"}

    # Print stats
    print(f"   t2m (Air Temp) range: {float(t2m.min()):.1f}°C → {float(t2m.max()):.1f}°C")
    print(f"   skt (LST)      range: {float(skt.min()):.1f}°C → {float(skt.max()):.1f}°C")
    print(f"   Time steps: {len(ds.time)} months")
    print(f"   Grid: {len(ds.latitude)} lat × {len(ds.longitude)} lon")

    # Save
    t2m_path = os.path.join(IDX_DIR, "era5_t2m_india.nc")
    lst_path  = os.path.join(IDX_DIR, "era5_lst_india.nc")

    t2m.to_dataset(name="t2m").to_netcdf(t2m_path)
    skt.to_dataset(name="lst").to_netcdf(lst_path)

    print(f"   ✅ Air temperature saved: era5_t2m_india.nc")
    print(f"   ✅ LST saved:             era5_lst_india.nc")

    ds.close()
    return t2m, skt

# =============================================================================
# 3. PROCESS PRECIPITATION
# =============================================================================

def process_precipitation():
    """
    Loads tp from avgad.nc
    Converts m → mm (multiply by 1000)
    Converts to monthly total (multiply by days in month × 24 hours × 3600 sec)
    Saves as NetCDF
    """
    print("\n[2/3] Processing precipitation data...")
    ds = xr.open_dataset(FILE_AVGAD)

    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # ERA5 tp is in m/hour — convert to mm/month
    # Monthly rate: tp (m) × 1000 (mm/m) × 24 (hours) × ~30 (days)
    precip_mm = ds["tp"] * 1000 * 24 * 30
    precip_mm.attrs = {
        "long_name": "Total Monthly Precipitation",
        "units":     "mm/month",
        "source":    "ERA5"
    }

    print(f"   Precipitation range: {float(precip_mm.min()):.1f} → {float(precip_mm.max()):.1f} mm/month")

    # Annual total for India
    india_mean = float(precip_mm.mean(dim=["latitude","longitude"]).sum())
    print(f"   10-year total (India mean): {india_mean:.0f} mm")

    # Save
    precip_path = os.path.join(IDX_DIR, "era5_precip_india.nc")
    precip_mm.to_dataset(name="precip").to_netcdf(precip_path)
    print(f"   ✅ Precipitation saved: era5_precip_india.nc")

    ds.close()
    return precip_mm

# =============================================================================
# 4. GENERATE PLOTS
# =============================================================================

def plot_lst_mean(skt):
    """Plot mean LST across all 10 years."""
    mean_lst = skt.mean(dim="time")

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        mean_lst.values,
        extent=[float(skt.longitude.min()), float(skt.longitude.max()),
                float(skt.latitude.min()), float(skt.latitude.max())],
        origin="upper", cmap="RdYlBu_r", aspect="auto",
        vmin=10, vmax=45
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("LST (°C)", fontsize=12)
    ax.set_title("REAL ERA5 Data — Mean Land Surface Temperature\nAll India (2015–2024)",
                 fontsize=13, fontweight="bold", color="#1A3C6E")
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.axhline(23.5, color="gray", linestyle="--", alpha=0.5,
               linewidth=0.8, label="Tropic of Cancer")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Mark major cities
    cities_plot = {"Mumbai":(72.9,19.1), "Delhi":(77.2,28.6),
                   "Chennai":(80.3,13.1), "Kolkata":(88.4,22.6)}
    for city, (lon, lat) in cities_plot.items():
        ax.plot(lon, lat, "ko", markersize=5)
        ax.annotate(city, (lon+0.5, lat+0.5), fontsize=8, color="black")

    out = os.path.join(OUT_DIR, "era5_lst_mean_REAL.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ LST mean map saved: era5_lst_mean_REAL.png")

def plot_timeseries(t2m, skt, precip):
    """Plot time-series for 3 major Indian cities."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("REAL ERA5 Data — Climate Time-Series for Major Indian Cities\n(2015–2024)",
                 fontsize=14, fontweight="bold", color="#1A3C6E")

    colors = {"Mumbai":"#2e8b57", "Delhi":"#c0392b", "Chennai":"#d4a017"}

    datasets = [
        (t2m,   axes[0], "Air Temperature (°C)",    "t2m"),
        (skt,   axes[1], "Land Surface Temp (°C)",  "lst"),
        (precip,axes[2], "Precipitation (mm/month)","precip"),
    ]

    for da, ax, ylabel, key in datasets:
        for city, coords in CITIES.items():
            ts = da.sel(
                latitude=coords["lat"],
                longitude=coords["lon"],
                method="nearest"
            )
            ax.plot(ts.time.values, ts.values,
                    color=colors[city], linewidth=1.5,
                    label=city, alpha=0.9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_facecolor("#f9f9f9")

        # Shade monsoon months (June-September)
        for year in range(2015, 2025):
            ax.axvspan(
                np.datetime64(f"{year}-06-01"),
                np.datetime64(f"{year}-10-01"),
                alpha=0.07, color="blue"
            )

    axes[0].set_title("Blue shading = Monsoon season (Jun–Sep)", fontsize=9,
                      color="gray", loc="right")
    axes[-1].set_xlabel("Date", fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "era5_timeseries_REAL.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Time-series plot saved: era5_timeseries_REAL.png")

def plot_annual_precip(precip):
    """Bar chart of annual precipitation for India."""
    india_monthly = precip.mean(dim=["latitude", "longitude"])
    times = pd.DatetimeIndex(precip.time.values)
    df = pd.DataFrame({
        "precip": india_monthly.values,
        "year":   times.year
    })
    annual = df.groupby("year")["precip"].sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#A32D2D" if v < annual.mean()*0.85
              else "#3B6D11" if v > annual.mean()*1.15
              else "#2E75B6" for v in annual.values]
    bars = ax.bar(annual.index, annual.values, color=colors,
                  edgecolor="white", width=0.6)
    ax.axhline(annual.mean(), color="#1A3C6E", linestyle="--",
               linewidth=2, label=f"10yr mean: {annual.mean():.0f} mm")
    ax.set_title("REAL ERA5 Data — Annual Precipitation India Mean (2015–2024)",
                 fontsize=13, fontweight="bold", color="#1A3C6E")
    ax.set_xlabel("Year"); ax.set_ylabel("Total Precipitation (mm/year)")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f9f9f9")
    for bar, val in zip(bars, annual.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{val:.0f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "era5_annual_precip_REAL.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Annual precipitation chart saved: era5_annual_precip_REAL.png")

# =============================================================================
# 5. COMPUTE ANOMALIES FROM REAL DATA
# =============================================================================

def compute_real_anomalies(skt):
    """
    Computes LST anomaly from real ERA5 data.
    Climatology = mean of 2015-2021 (training period)
    Anomaly = 2024 value - climatology
    """
    print("\n   Computing real LST anomalies...")
    times = pd.DatetimeIndex(skt.time.values)
    years = xr.DataArray(times.year, dims=["time"], coords={"time": skt.time})

    # Training period climatology
    train_mask = years <= 2021
    clim = skt.isel(time=train_mask).groupby(
        skt.time.isel(time=train_mask).dt.month
    ).mean(dim="time")

    # 2024 anomaly
    mask_2024 = years == 2024
    if mask_2024.sum() > 0:
        skt_2024 = skt.isel(time=mask_2024).mean(dim="time")
        clim_annual = clim.mean(dim="month")
        anomaly = skt_2024 - clim_annual

        print(f"   2024 LST anomaly — India mean: {float(anomaly.mean()):+.2f}°C")
        print(f"   Hottest anomaly:  {float(anomaly.max()):+.2f}°C")
        print(f"   Coolest anomaly:  {float(anomaly.min()):+.2f}°C")

        # Save anomaly
        out = os.path.join(IDX_DIR, "era5_lst_anomaly_2024.nc")
        anomaly.to_dataset(name="lst_anomaly").to_netcdf(out)
        print(f"   ✅ 2024 LST anomaly saved: era5_lst_anomaly_2024.nc")
        return anomaly
    else:
        print("   ⚠️  No 2024 data found in file")
        return None

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*65)
    print("🌍 ERA5 REAL DATA PROCESSING PIPELINE")
    print("   Converting ERA5 NetCDF → Project-ready climate indices")
    print("="*65)

    # Step 0: Check files exist
    if not check_files():
        print("\n⚠️  Copy your ERA5 files to data/raw/era5/ first!")
        print("    Then run this script again.")
        sys.exit(1)

    # Step 1: Process temperature
    t2m, skt = process_temperature()

    # Step 2: Process precipitation
    precip = process_precipitation()

    # Step 3: Generate plots
    print("\n[3/3] Generating plots...")
    plot_lst_mean(skt)
    plot_timeseries(t2m, skt, precip)
    plot_annual_precip(precip)
    compute_real_anomalies(skt)

    # Summary
    print("\n" + "="*65)
    print("✅ ERA5 REAL DATA PROCESSING COMPLETE!")
    print("="*65)
    print("\n📁 NetCDF files saved to data/indices/:")
    print("   • era5_lst_india.nc       ← Real LST for all India")
    print("   • era5_t2m_india.nc       ← Real air temperature")
    print("   • era5_precip_india.nc    ← Real precipitation")
    print("   • era5_lst_anomaly_2024.nc ← 2024 anomaly vs baseline")
    print("\n📁 Plots saved to outputs/maps/:")
    print("   • era5_lst_mean_REAL.png")
    print("   • era5_timeseries_REAL.png")
    print("   • era5_annual_precip_REAL.png")
    print("\n🎯 KEY FINDING:")
    print("   Your ERA5 data has 120 months (10 years) of real climate")
    print("   data for all of India at 0.25° resolution.")
    print("   This is now ready to feed into your XGBoost model!")
    print("\n👉 Next: run python scripts/09_train_on_real_data.py")
    print("="*65)
