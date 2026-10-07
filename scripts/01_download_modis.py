# 01_download_modis.py  (fixed)
import ee, geemap, os, time

PROJECT_ID = "secret-opus-464414-d7"
SCALE = 5000   # metres. 5 km keeps each file under the 50 MB limit

ee.Initialize(project=PROJECT_ID)
print("Earth Engine initialized")

india = ee.Geometry.Rectangle([68.1, 8.4, 97.4, 37.6])
ROOT = r"C:\climate_project\data\raw\modis"

def export(img, path):
    """Download and only report success if the file really exists."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        geemap.ee_export_image(img, filename=path, scale=SCALE,
                               region=india, file_per_band=False)
    except Exception as e:
        print("   error:", e)
    ok = os.path.exists(path) and os.path.getsize(path) > 0
    return "ok" if ok else "FAIL"

ok = fail = 0
for year in range(2015, 2025):
    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01"

        veg = (ee.ImageCollection("MODIS/061/MOD13A3")
               .filterDate(start, end).select(["NDVI", "EVI"]).mean())
        lst = (ee.ImageCollection("MODIS/061/MOD11A2")
               .filterDate(start, end).select("LST_Day_1km").mean())

        jobs = {
            "ndvi": veg.select("NDVI").multiply(0.0001).clip(india),
            "evi":  veg.select("EVI").multiply(0.0001).clip(india),
            "lst":  lst.multiply(0.02).subtract(273.15).clip(india),
        }
        for name, img in jobs.items():
            path = os.path.join(ROOT, name, str(year),
                                f"{name.upper()}_{year}_{month:02d}.tif")
            r = export(img, path)
            print(f"{name:5s} {year}-{month:02d}: {r}")
            ok += r in ("ok", "skip")
            fail += r == "FAIL"
        time.sleep(0.5)

print(f"\nFinished. Success: {ok} | Failed: {fail} (expected 360 total)")