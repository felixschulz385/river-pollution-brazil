from pathlib import Path
import earthkit.data as ekd

path = Path("data/climate/raw/era5_land_daily/era5_land_daily_1985_11.grib")
print(f"exists={path.exists()} size_bytes={path.stat().st_size if path.exists() else None}")

fl = ekd.from_source("file", str(path))
try:
    meta = fl.ls()
    print("columns=", list(meta.columns))
    print("rows=", len(meta), "cols=", len(meta.columns))
    print("head=")
    print(meta.head(10).to_string())

    arr = fl.to_numpy()
    print("numpy_shape=", arr.shape)

    lat = fl.data(keys="lat")
    lon = fl.data(keys="lon")
    print("lat_shape=", getattr(lat, "shape", None))
    print("lon_shape=", getattr(lon, "shape", None))
finally:
    close = getattr(fl, "close", None)
    if callable(close):
        close()