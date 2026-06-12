"""
Generate test stores for spatial, geolocation, and broken_refs conventions.

Produces:
  spatial_test.zarr      - spatial convention, single array + group-level inheritance
  geolocation_test.zarr  - geolocation convention, curvilinear grid with lat/lon arrays
  broken_refs.zarr       - store with unresolvable references for error handling tests
"""
import numpy as np
import zarr
import os, shutil
from pathlib import Path

STORES = Path("/Users/patrickvanlaake/zarr/backend_test_stores")
STORES.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

SPATIAL_UUID = "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"
GEO_UUID     = "bb9ee930-8c60-4c47-ad6b-8daa558987ed"
REF_UUID     = "d89b30cf-ed8c-43d5-9a16-b492f0cd8786"
PROJ_UUID    = "f17cb550-5864-4468-aeb7-f3180cfb622f"

def cmo(uuid, name, schema_url=None):
    d = {"uuid": uuid, "name": name}
    if schema_url:
        d["schema_url"] = schema_url
    return d

# ===========================================================================
# spatial_test.zarr
# ===========================================================================
SP = STORES / "spatial_test.zarr"
if SP.exists(): shutil.rmtree(SP)
sp = zarr.open(str(SP), mode="w")
sp.attrs.update({"title": "spatial convention test store"})

# /single — array with all spatial: attrs at array level
# Sentinel-2-like: 10m pixels, 100x100 subset, north-up
g_single = sp.require_group("single")
data_single = rng.random((100, 100)).astype("f4")
arr_single = g_single.create_array(
    "band",
    data=data_single,
    chunks=(100, 100),
    dimension_names=["y", "x"]
)
arr_single.attrs.update({
    "zarr_conventions": [
        cmo(PROJ_UUID, "proj:"),
        cmo(SPATIAL_UUID, "spatial")
    ],
    "proj:code": "EPSG:32632",
    "spatial:dimensions": ["y", "x"],
    "spatial:transform": [10.0, 0.0, 300000.0, 0.0, -10.0, 5200000.0],
    "spatial:shape": [100, 100],
    "spatial:bbox": [300000.0, 5199000.0, 301000.0, 5200000.0],
    "spatial:registration": "pixel",
    "long_name": "surface reflectance",
})

# /shared — group-level spatial: attrs, two arrays inherit them
g_shared = sp.require_group("shared")
g_shared.attrs.update({
    "zarr_conventions": [
        cmo(SPATIAL_UUID, "spatial")
    ],
    "spatial:dimensions": ["y", "x"],
    "spatial:transform": [20.0, 0.0, 300000.0, 0.0, -20.0, 5200000.0],
    "spatial:shape": [50, 50],
    "spatial:bbox": [300000.0, 5199000.0, 301000.0, 5200000.0],
    "spatial:registration": "pixel",
})
for band in ("b05", "b06"):
    data = rng.random((50, 50)).astype("f4")
    arr = g_shared.create_array(band, data=data, chunks=(50, 50),
                                dimension_names=["y", "x"])
    arr.attrs.update({"long_name": f"band {band}"})

# /node_reg — node registration (DEM-style)
g_node = sp.require_group("node_reg")
data_dem = (200.0 + 800.0 * rng.random((30, 30))).astype("f4")
arr_dem = g_node.create_array("dem", data=data_dem, chunks=(30, 30),
                               dimension_names=["y", "x"])
arr_dem.attrs.update({
    "zarr_conventions": [cmo(SPATIAL_UUID, "spatial")],
    "spatial:dimensions": ["y", "x"],
    "spatial:transform": [0.000277778, 0.0, -10.0, 0.0, -0.000277778, 50.0],
    "spatial:shape": [30, 30],
    "spatial:registration": "node",
    "long_name": "digital elevation model",
    "units": "m",
})

print(f"Written {SP}")

# ===========================================================================
# geolocation_test.zarr
# ===========================================================================
GL = STORES / "geolocation_test.zarr"
if GL.exists(): shutil.rmtree(GL)
gl = zarr.open(str(GL), mode="w")
gl.attrs.update({"title": "geolocation convention test store"})

N_ROW, N_COL = 40, 50

# Synthetic curvilinear grid (CORDEX-style rotated pole)
rlon_vals = np.linspace(-28.375, -28.375 + 0.44 * N_COL, N_COL)
rlat_vals = np.linspace(-23.375, -23.375 + 0.44 * N_ROW, N_ROW)
RLON, RLAT = np.meshgrid(rlon_vals, rlat_vals)
lon2d = (RLON + 0.1 * np.sin(np.pi * RLAT / 30)).astype("f8")
lat2d = (RLAT + 0.1 * np.cos(np.pi * RLON / 30)).astype("f8")

# Geolocation arrays in /coords group
g_coords = gl.require_group("coords")
lon_arr = g_coords.create_array("longitude", data=lon2d,
                                 chunks=(N_ROW, N_COL),
                                 dimension_names=["rlat", "rlon"])
lon_arr.attrs.update({"standard_name": "longitude", "units": "degrees_east"})

lat_arr = g_coords.create_array("latitude", data=lat2d,
                                 chunks=(N_ROW, N_COL),
                                 dimension_names=["rlat", "rlon"])
lat_arr.attrs.update({"standard_name": "latitude", "units": "degrees_north"})

# /data group — precipitation array with geolocation references
g_data = gl.require_group("data")
pr_data = (0.001 * rng.standard_normal((N_ROW, N_COL))).astype("f4")
pr_arr = g_data.create_array("pr", data=pr_data, chunks=(N_ROW, N_COL),
                               dimension_names=["rlat", "rlon"])
pr_arr.attrs.update({
    "zarr_conventions": [
        cmo(GEO_UUID, "geolocation",
            "https://raw.githubusercontent.com/R-CF/zarr_convention_geolocation/main/schema.json"),
        cmo(REF_UUID, "ref",
            "https://raw.githubusercontent.com/R-CF/zarr_convention_ref/main/schema.json"),
    ],
    "geolocation": {
        "geodetic": {
            "x": {"ref": {"node": "../coords/longitude"}},
            "y": {"ref": {"node": "../coords/latitude"}},
            "crs": {"proj:code": "EPSG:4326"}
        }
    },
    "long_name": "precipitation flux",
    "units": "kg m-2 s-1",
})

# /data/both — array with both geodetic and planar geolocation
# Add simple planar arrays (UTM-like)
utm_x = g_coords.create_array("utm_x", data=(lon2d * 111320).astype("f8"),
                                chunks=(N_ROW, N_COL),
                                dimension_names=["rlat", "rlon"])
utm_x.attrs.update({"units": "m"})
utm_y = g_coords.create_array("utm_y", data=(lat2d * 110540).astype("f8"),
                                chunks=(N_ROW, N_COL),
                                dimension_names=["rlat", "rlon"])
utm_y.attrs.update({"units": "m"})

temp_data = (280.0 + 10.0 * rng.standard_normal((N_ROW, N_COL))).astype("f4")
temp_arr = g_data.create_array("temp", data=temp_data, chunks=(N_ROW, N_COL),
                                dimension_names=["rlat", "rlon"])
temp_arr.attrs.update({
    "zarr_conventions": [
        cmo(GEO_UUID, "geolocation",
            "https://raw.githubusercontent.com/R-CF/zarr_convention_geolocation/main/schema.json"),
        cmo(REF_UUID, "ref",
            "https://raw.githubusercontent.com/R-CF/zarr_convention_ref/main/schema.json"),
    ],
    "geolocation": {
        "geodetic": {
            "x": {"ref": {"node": "../coords/longitude"}},
            "y": {"ref": {"node": "../coords/latitude"}},
            "crs": {"proj:code": "EPSG:4326"}
        },
        "planar": {
            "x": {"ref": {"node": "../coords/utm_x"}},
            "y": {"ref": {"node": "../coords/utm_y"}},
        }
    },
    "long_name": "air temperature",
    "units": "K",
})

print(f"Written {GL}")

# ===========================================================================
# broken_refs.zarr
# ===========================================================================
BR = STORES / "broken_refs.zarr"
if BR.exists(): shutil.rmtree(BR)
br = zarr.open(str(BR), mode="w")
br.attrs.update({"title": "broken references test store"})

CS_UUID = "e4dbf0b7-7a00-4ce6-b23e-484292014ab4"
REF_UUID = "d89b30cf-ed8c-43d5-9a16-b492f0cd8786"


# Array with a reference to a non-existent node
g_bad = br.require_group("bad")
data_bad = rng.random((5, 5)).astype("f4")
arr_bad = g_bad.create_array("var", data=data_bad, chunks=(5, 5),
                              dimension_names=["y", "x"])
arr_bad.attrs.update({
    "zarr_conventions": [
        {"uuid": CS_UUID, "name": "cs"},
        {"uuid": REF_UUID, "name": "ref"},
    ],
    "cs": {
        "crs": [{
            "axes": {
                "x": {
                    "abbreviation": "X",
                    "coordinates": [{
                        "direction": "east",
                        "unit": "m",
                        "values": {
                            "external": {
                                "ref": {"node": "../nonexistent/coords"}
                            }
                        }
                    }]
                },
                "y": {
                    "abbreviation": "Y",
                    "coordinates": [{
                        "direction": "north",
                        "unit": "m",
                        "values": {"regular": [0.0, 1000.0]}
                    }]
                }
            }
        }]
    }
})

# Array with a malformed ref path
arr_malformed = g_bad.create_array(
    "malformed", data=rng.random((5, 5)).astype("f4"),
    chunks=(5, 5), dimension_names=["y", "x"]
)
arr_malformed.attrs.update({
    "zarr_conventions": [
        {"uuid": CS_UUID, "name": "cs"},
        {"uuid": REF_UUID, "name": "ref"},
    ],
    "cs": {
        "crs": [{
            "axes": {
                "x": {
                    "abbreviation": "X",
                    "coordinates": [{
                        "direction": "east",
                        "unit": "m",
                        "values": {
                            "external": {
                                "ref": {}  # missing "node" field
                            }
                        }
                    }]
                }
            }
        }]
    }
})

print(f"Written {BR}")
