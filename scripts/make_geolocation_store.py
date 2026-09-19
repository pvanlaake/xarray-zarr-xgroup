"""
Generate geolocation_test.zarr for xarray-zarr-xgroup tests.
Run from project root: python scripts/make_geolocation_store.py

All ref paths are relative to the referencing array (not its group).
"""
import numpy as np
import zarr
import os, shutil
from pathlib import Path

STORE = Path(__file__).parent.parent / "zarr_xgroup" / "tests" / "stores" / "geolocation_test.zarr"
if STORE.exists(): shutil.rmtree(STORE)
STORE.parent.mkdir(parents=True, exist_ok=True)

CS_UUID  = "e4dbf0b7-7a00-4ce6-b23e-484292014ab4"
GEO_UUID = "bb9ee930-8c60-4c47-ad6b-8daa558987ed"
REF_UUID = "d89b30cf-ed8c-43d5-9a16-b492f0cd8786"

def cmo(uuid, name, schema_url=None):
    d = {"uuid": uuid, "name": name}
    if schema_url: d["schema_url"] = schema_url
    return d

rng = np.random.default_rng(42)
N_ROW, N_COL = 40, 50

root = zarr.open(str(STORE), mode="w")
root.attrs.update({"title": "geolocation convention test store"})

# /coords — geolocation arrays
g_coords = root.require_group("coords")
rlon_vals = np.linspace(-28.375, -28.375 + 0.44*N_COL, N_COL)
rlat_vals = np.linspace(-23.375, -23.375 + 0.44*N_ROW, N_ROW)
RLON, RLAT = np.meshgrid(rlon_vals, rlat_vals)
lon2d = (RLON + 0.1*np.sin(np.pi*RLAT/30)).astype("f8")
lat2d = (RLAT + 0.1*np.cos(np.pi*RLON/30)).astype("f8")

for name, data, attrs in [
    ("longitude", lon2d,                      {"standard_name": "longitude", "units": "degrees_east"}),
    ("latitude",  lat2d,                      {"standard_name": "latitude",  "units": "degrees_north"}),
    ("utm_x",     (lon2d*111320).astype("f8"), {"units": "m"}),
    ("utm_y",     (lat2d*110540).astype("f8"), {"units": "m"}),
]:
    a = g_coords.create_array(name, data=data, chunks=(N_ROW, N_COL), dimension_names=["rlat", "rlon"])
    a.attrs.update(attrs)

def rotated_crs(geodetic_only=True):
    # From /data/pr or /data/temp: ../../coords/longitude → /coords/longitude ✓
    geo = {
        "geodetic": {
            "x": {"ref": {"node": "../../coords/longitude"}},
            "y": {"ref": {"node": "../../coords/latitude"}},
            "crs": {"proj:code": "EPSG:4326"}
        }
    }
    if not geodetic_only:
        geo["planar"] = {
            "x": {"ref": {"node": "../../coords/utm_x"}},
            "y": {"ref": {"node": "../../coords/utm_y"}},
        }
    return {
        "type": "planar",
        "name": "rotated pole grid",
        "axes": {
            "rlon": {"abbreviation": "X", "coordinates": [{"direction": "east",  "unit": "degrees", "values": {"regular": [-28.375, 0.44]}}]},
            "rlat": {"abbreviation": "Y", "coordinates": [{"direction": "north", "unit": "degrees", "values": {"regular": [-23.375, 0.44]}}]}
        },
        "geolocation": geo
    }

g_data = root.require_group("data")

# pr — geodetic geolocation only
pr_arr = g_data.create_array("pr",
    data=(0.001*rng.standard_normal((N_ROW, N_COL))).astype("f4"),
    chunks=(N_ROW, N_COL), dimension_names=["rlat", "rlon"])
pr_arr.attrs.update({
    "zarr_conventions": [
        cmo(CS_UUID,  "cs",          "https://raw.githubusercontent.com/R-CF/zarr_convention_cs/main/schema.json"),
        cmo(GEO_UUID, "geolocation", "https://raw.githubusercontent.com/R-CF/zarr_convention_geolocation/main/schema.json"),
        cmo(REF_UUID, "ref",         "https://raw.githubusercontent.com/R-CF/zarr_convention_ref/main/schema.json"),
    ],
    "cs": {"crs": [rotated_crs(geodetic_only=True)]},
    "long_name": "precipitation flux", "units": "kg m-2 s-1",
})

# temp — both geodetic and planar geolocation
temp_arr = g_data.create_array("temp",
    data=(280.0+10.0*rng.standard_normal((N_ROW, N_COL))).astype("f4"),
    chunks=(N_ROW, N_COL), dimension_names=["rlat", "rlon"])
temp_arr.attrs.update({
    "zarr_conventions": [
        cmo(CS_UUID,  "cs",          "https://raw.githubusercontent.com/R-CF/zarr_convention_cs/main/schema.json"),
        cmo(GEO_UUID, "geolocation", "https://raw.githubusercontent.com/R-CF/zarr_convention_geolocation/main/schema.json"),
        cmo(REF_UUID, "ref",         "https://raw.githubusercontent.com/R-CF/zarr_convention_ref/main/schema.json"),
    ],
    "cs": {"crs": [rotated_crs(geodetic_only=False)]},
    "long_name": "air temperature", "units": "K",
})

print(f"Written {STORE}")
def inventory(grp, indent=0):
    for name in sorted(grp.array_keys()):
        print("  "*indent + f"  [{name}]  {grp[name].shape}")
    for name in sorted(grp.group_keys()):
        print("  "*indent + f"  /{name}/")
        inventory(grp[name], indent+1)
print("\n=== Inventory ===")
inventory(root)
