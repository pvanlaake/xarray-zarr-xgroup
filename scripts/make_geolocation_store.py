"""
Generate geolocation_test.zarr — a test store for the geolocation service
convention, with cs as the principal convention on all data arrays.

Group layout:
  /coords/          geolocation arrays (longitude, latitude, utm_x, utm_y)
  /data/pr          precipitation — cs + geodetic geolocation only
  /data/temp        temperature   — cs + geodetic and planar geolocation
"""
import numpy as np
import zarr
import os, shutil
from pathlib import Path

STORE = Path(__file__).parent.parent / "zarr_xgroup" / "tests" / "stores" / "geolocation_test.zarr"
if STORE.exists(): shutil.rmtree(STORE)
STORE.parent.mkdir(exist_ok=True)

CS_UUID  = "e4dbf0b7-7a00-4ce6-b23e-484292014ab4"
GEO_UUID = "bb9ee930-8c60-4c47-ad6b-8daa558987ed"
REF_UUID = "d89b30cf-ed8c-43d5-9a16-b492f0cd8786"

def cmo(uuid, name, schema_url=None):
    d = {"uuid": uuid, "name": name}
    if schema_url:
        d["schema_url"] = schema_url
    return d

rng = np.random.default_rng(42)
N_ROW, N_COL = 40, 50

root = zarr.open(str(STORE), mode="w")
root.attrs.update({"title": "geolocation convention test store"})

# /coords — geolocation arrays
g_coords = root.require_group("coords")

rlon_vals = np.linspace(-28.375, -28.375 + 0.44 * N_COL, N_COL)
rlat_vals = np.linspace(-23.375, -23.375 + 0.44 * N_ROW, N_ROW)
RLON, RLAT = np.meshgrid(rlon_vals, rlat_vals)
lon2d = (RLON + 0.1 * np.sin(np.pi * RLAT / 30)).astype("f8")
lat2d = (RLAT + 0.1 * np.cos(np.pi * RLON / 30)).astype("f8")

lon_arr = g_coords.create_array("longitude", data=lon2d,
    chunks=(N_ROW, N_COL), dimension_names=["rlat", "rlon"])
lon_arr.attrs.update({"standard_name": "longitude", "units": "degrees_east"})

lat_arr = g_coords.create_array("latitude", data=lat2d,
    chunks=(N_ROW, N_COL), dimension_names=["rlat", "rlon"])
lat_arr.attrs.update({"standard_name": "latitude", "units": "degrees_north"})

utm_x = g_coords.create_array("utm_x",
    data=(lon2d * 111320).astype("f8"),
    chunks=(N_ROW, N_COL), dimension_names=["rlat", "rlon"])
utm_x.attrs.update({"units": "m"})

utm_y = g_coords.create_array("utm_y",
    data=(lat2d * 110540).astype("f8"),
    chunks=(N_ROW, N_COL), dimension_names=["rlat", "rlon"])
utm_y.attrs.update({"units": "m"})

# /data group
g_data = root.require_group("data")

# pr — geodetic geolocation only
pr_data = (0.001 * rng.standard_normal((N_ROW, N_COL))).astype("f4")
pr_arr = g_data.create_array("pr", data=pr_data,
    chunks=(N_ROW, N_COL), dimension_names=["rlat", "rlon"])
pr_arr.attrs.update({
    "zarr_conventions": [
        cmo(CS_UUID, "cs",
            "https://raw.githubusercontent.com/R-CF/zarr_convention_cs/main/schema.json"),
        cmo(GEO_UUID, "geolocation",
            "https://raw.githubusercontent.com/R-CF/zarr_convention_geolocation/main/schema.json"),
        cmo(REF_UUID, "ref",
            "https://raw.githubusercontent.com/R-CF/zarr_convention_ref/main/schema.json"),
    ],
    "cs": {
        "crs": [{
            "name": "rotated pole grid",
            "axes": {
                "rlon": {
                    "abbreviation": "X",
                    "coordinates": [{
                        "direction": "east",
                        "unit": "degrees",
                        "values": {"regular": [-28.375, 0.44]}
                    }]
                },
                "rlat": {
                    "abbreviation": "Y",
                    "coordinates": [{
                        "direction": "north",
                        "unit": "degrees",
                        "values": {"regular": [-23.375, 0.44]}
                    }]
                }
            },
            "geolocation": {
                "geodetic": {
                    "x": {"ref": {"node": "../coords/longitude"}},
                    "y": {"ref": {"node": "../coords/latitude"}},
                    "crs": {"proj:code": "EPSG:4326"}
                }
            }
        }]
    },
    "long_name": "precipitation flux",
    "units": "kg m-2 s-1",
})

# temp — both geodetic and planar geolocation in the same geolocation object
temp_data = (280.0 + 10.0 * rng.standard_normal((N_ROW, N_COL))).astype("f4")
temp_arr = g_data.create_array("temp", data=temp_data,
    chunks=(N_ROW, N_COL), dimension_names=["rlat", "rlon"])
temp_arr.attrs.update({
    "zarr_conventions": [
        cmo(CS_UUID, "cs",
            "https://raw.githubusercontent.com/R-CF/zarr_convention_cs/main/schema.json"),
        cmo(GEO_UUID, "geolocation",
            "https://raw.githubusercontent.com/R-CF/zarr_convention_geolocation/main/schema.json"),
        cmo(REF_UUID, "ref",
            "https://raw.githubusercontent.com/R-CF/zarr_convention_ref/main/schema.json"),
    ],
    "cs": {
        "crs": [{
            "name": "rotated pole grid",
            "axes": {
                "rlon": {
                    "abbreviation": "X",
                    "coordinates": [{
                        "direction": "east",
                        "unit": "degrees",
                        "values": {"regular": [-28.375, 0.44]}
                    }]
                },
                "rlat": {
                    "abbreviation": "Y",
                    "coordinates": [{
                        "direction": "north",
                        "unit": "degrees",
                        "values": {"regular": [-23.375, 0.44]}
                    }]
                }
            },
            "geolocation": {
                "geodetic": {
                    "x": {"ref": {"node": "../coords/longitude"}},
                    "y": {"ref": {"node": "../coords/latitude"}},
                    "crs": {"proj:code": "EPSG:4326"}
                },
                "planar": {
                    "x": {"ref": {"node": "../coords/utm_x"}},
                    "y": {"ref": {"node": "../coords/utm_y"}}
                }
            }
        }]
    },
    "long_name": "air temperature",
    "units": "K",
})

print(f"Written {STORE}")

def inventory(grp, indent=0):
    pad = "  " * indent
    for name in sorted(grp.array_keys()):
        arr = grp[name]
        print(f"{pad}  [{name}]  shape={arr.shape}  dims={arr.metadata.dimension_names}")
    for name in sorted(grp.group_keys()):
        print(f"{pad}  /{name}/")
        inventory(grp[name], indent+1)

print("\n=== Inventory ===")
inventory(root)
