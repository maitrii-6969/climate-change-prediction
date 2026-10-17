# inspect_modis.py - read-only check of the downloaded MODIS GeoTIFFs. Changes nothing.
import os, sys, re, glob, collections
import numpy as np
try:
    import rasterio
except ImportError:
    sys.exit("Run first:  pip install rasterio")

ROOT = os.environ.get("MODIS_DIR", r"C:\climate_project\data\raw\modis")
files = sorted(glob.glob(os.path.join(ROOT, "**", "*.tif"), recursive=True))
print("total .tif files:", len(files))
print("sub-folders:", sorted({os.path.relpath(os.path.dirname(f), ROOT) for f in files}))
print("first 3 names:", [os.path.basename(f) for f in files[:3]])
print("last  3 names:", [os.path.basename(f) for f in files[-3:]])

# group by variable word in the file name
groups = collections.defaultdict(list)
for f in files:
    n = os.path.basename(f).upper()
    key = "NDVI" if "NDVI" in n else "EVI" if "EVI" in n else "LST" if "LST" in n else "OTHER"
    groups[key].append(f)
print("counts by name:", {k: len(v) for k, v in groups.items()})

for key, fl in groups.items():
    f = fl[len(fl) // 2]
    with rasterio.open(f) as r:
        a = r.read(1).astype("float64")
        nod = r.nodata
        v = a[np.isfinite(a)]
        if nod is not None: v = v[v != nod]
        print(f"\n[{key}] {os.path.basename(f)}")
        print("  shape", r.height, "x", r.width, "| bands", r.count, "| dtype", r.dtypes[0],
              "| crs", r.crs, "| nodata", nod, "| band names", r.descriptions)
        print("  pixel size", r.res, "| bounds", tuple(round(x, 2) for x in r.bounds))
        if v.size:
            print("  valid pixels %d of %d | min %.3f  mean %.3f  max %.3f" % (v.size, a.size, v.min(), v.mean(), v.max()))
        else:
            print("  NO valid pixels in this file")

# any files that are tiny/empty?
small = [os.path.basename(f) for f in files if os.path.getsize(f) < 2000]
print("\nfiles under 2 KB (likely failed):", len(small), small[:5])
