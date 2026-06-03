"""
Generate a minimal cs-convention test store for xarray-zarr-xgroup.
Run this script to produce zarr_xgroup/tests/stores/cs_test.zarr.

Usage:
    python make_cs_test_store.py

Exercises:
  - Regular inline spatial coordinates
  - External time coordinate array via ref
  - Cross-group crs reference (shared CRS in parent group)  
  - Geolocation arrays via ref (CORDEX-style curvilinear grid)
  - Parametric vertical coordinates via ref terms (ROMS-style)
"""
import numpy as np
import zarr
import os, shutil
from pathlib import Path

STORE_PATH = Path(__file__).parent / "zarr_xgroup" / "tests" / "stores" / "cs_test.zarr"

if os.path.exists(STORE_PATH):
    shutil.rmtree(STORE_PATH)
STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

CS_CMO = {
    "schema_url": "https://raw.githubusercontent.com/R-CF/zarr_convention_cs/main/schema.json",
    "uuid": "e4dbf0b7-7a00-4ce6-b23e-484292014ab4",
    "name": "cs",
}
REF_CMO = {
    "schema_url": "https://raw.githubusercontent.com/R-CF/zarr_convention_ref/main/schema.json",
    "uuid": "d89b30cf-ed8c-43d5-9a16-b492f0cd8786",
    "name": "ref",
}
GEO_CMO = {
    "schema_url": "https://raw.githubusercontent.com/R-CF/zarr_convention_geolocation/main/schema.json",
    "name": "geolocation",
}

root = zarr.open(str(STORE_PATH), mode="w")
root.attrs.update({
    "title": "cs-convention test store for xarray-zarr-xgroup",
    "comment": "Exercises all reference patterns in the cs convention.",
})

rng = np.random.default_rng(42)
N_TIME = 24
N_LAT, N_LON = 36, 72
N_RLAT, N_RLON = 20, 25
N_T, N_S, N_ETA, N_XI = N_TIME, 5, 12, 15

# /coords — shared CRS definitions and time coordinate arrays
g_coords = root.require_group("coords")
g_coords.attrs.update({
    "zarr_conventions": [CS_CMO, REF_CMO],
    "crs": {
        "WGS84": {
            "axes": {
                "lon": {"abbreviation": "X", "coordinates": [{"direction": "east", "unit": "degrees", "values": {"regular": [-179.75, 0.5]}}]},
                "lat": {"abbreviation": "Y", "coordinates": [{"direction": "north", "unit": "degrees", "values": {"regular": [-89.75, 0.5]}}]}
            },
            "id": {"proj:code": "EPSG:4326"}
        },
        "standard_calendar": {
            "axes": {
                "time": {
                    "abbreviation": "T",
                    "coordinates": [{
                        "direction": "future",
                        "time": {"unit": "days", "epoch": "1850-01-01", "calendar": "standard"},
                        "values": {"external": {"ref": {"node": "../coords/time"}}},
                        "boundaries": {"external": {"ref": {"node": "../coords/time_bnds"}}}
                    }]
                }
            }
        }
    }
})

time_data = np.array([365*y + 30*m + 15 for y in range(2) for m in range(12)], dtype="f8")
t = g_coords.create_array("time", data=time_data, chunks=(N_TIME,), dimension_names=["time"])
t.attrs.update({"long_name": "time"})
time_bnds = np.column_stack([time_data - 15, time_data + 15]).astype("f8")
g_coords.create_array("time_bnds", data=time_bnds, chunks=(N_TIME, 2), dimension_names=["time", "bounds"])

# /simple — regular grid, cross-group CRS refs
g_simple = root.require_group("simple")
temp_data = (15.0 + 10.0 * rng.standard_normal((N_TIME, N_LAT, N_LON))).astype("f4")
arr = g_simple.create_array("temperature", data=temp_data, chunks=(1, N_LAT, N_LON), dimension_names=["time", "lat", "lon"])
arr.attrs.update({
    "long_name": "air temperature", "units": "degrees_C",
    "zarr_conventions": [CS_CMO, REF_CMO],
    "cs": {"crs": [
        {"ref": {"node": "../coords", "attribute": "/attributes/crs/WGS84"}},
        {"ref": {"node": "../coords", "attribute": "/attributes/crs/standard_calendar"}}
    ]}
})

# /curvilinear — rotated grid with geolocation arrays
g_curv = root.require_group("curvilinear")
rlon_vals = np.linspace(-28.375, -28.375 + 0.11*N_RLON, N_RLON)
rlat_vals = np.linspace(-23.375, -23.375 + 0.11*N_RLAT, N_RLAT)
RLON, RLAT = np.meshgrid(rlon_vals, rlat_vals)
lon2d = (RLON + 0.05*np.sin(np.pi*RLAT/30)).astype("f8")
lat2d = (RLAT + 0.05*np.cos(np.pi*RLON/30)).astype("f8")
la = g_curv.create_array("lon", data=lon2d, chunks=(N_RLAT, N_RLON), dimension_names=["rlat", "rlon"])
la.attrs.update({"standard_name": "longitude", "units": "degrees_east"})
lo = g_curv.create_array("lat", data=lat2d, chunks=(N_RLAT, N_RLON), dimension_names=["rlat", "rlon"])
lo.attrs.update({"standard_name": "latitude", "units": "degrees_north"})
pr_data = (0.001*rng.standard_normal((N_TIME, N_RLAT, N_RLON))).astype("f4")
pr = g_curv.create_array("pr", data=pr_data, chunks=(1, N_RLAT, N_RLON), dimension_names=["time", "rlat", "rlon"])
pr.attrs.update({
    "long_name": "precipitation flux", "units": "kg m-2 s-1",
    "zarr_conventions": [CS_CMO, REF_CMO, GEO_CMO],
    "cs": {"crs": [
        {
            "name": "rotated pole grid",
            "axes": {
                "rlon": {"abbreviation": "X", "coordinates": [{"direction": "east", "unit": "degrees", "values": {"regular": [-28.375, 0.11]}}]},
                "rlat": {"abbreviation": "Y", "coordinates": [{"direction": "north", "unit": "degrees", "values": {"regular": [-23.375, 0.11]}}]}
            },
            "geolocation": {"geodetic": {
                "x": {"ref": {"node": "../curvilinear/lon"}},
                "y": {"ref": {"node": "../curvilinear/lat"}},
                "crs": {"proj:code": "EPSG:4326"}
            }}
        },
        {"ref": {"node": "../coords", "attribute": "/attributes/crs/standard_calendar"}}
    ]}
})

# /params — parametric term arrays
g_params = root.require_group("params")
s_rho_data = np.linspace(-1.0, 0.0, N_S, endpoint=False) + 0.5/N_S
g_params.create_array("s_rho", data=s_rho_data.astype("f8"), chunks=(N_S,), dimension_names=["s_rho"])
theta_s, theta_b = 6.0, 0.9
Cs_r = ((1-theta_b)*np.sinh(theta_s*s_rho_data)/np.sinh(theta_s)
        + theta_b*(np.tanh(theta_s*(s_rho_data+0.5))/(2*np.tanh(0.5*theta_s))-0.5))
g_params.create_array("Cs_r", data=Cs_r.astype("f8"), chunks=(N_S,), dimension_names=["s_rho"])
h_data = (200.0 + 800.0*rng.random((N_ETA, N_XI))).astype("f8")
g_params.create_array("h", data=h_data, chunks=(N_ETA, N_XI), dimension_names=["eta_rho", "xi_rho"])
g_params.create_array("hc", data=np.array(20.0, dtype="f8"), chunks=(), dimension_names=[])

# /ocean — 4D array with parametric vertical coords
g_ocean = root.require_group("ocean")
temp_ocean = (15.0 + 5.0*s_rho_data[np.newaxis,:,np.newaxis,np.newaxis]
              + 0.5*rng.standard_normal((N_T, N_S, N_ETA, N_XI))).astype("f4")
ot = g_ocean.create_array("temp", data=temp_ocean, chunks=(1, N_S, N_ETA, N_XI),
                           dimension_names=["time", "s_rho", "eta_rho", "xi_rho"])
ot.attrs.update({
    "long_name": "sea water potential temperature", "units": "degrees_C",
    "zarr_conventions": [CS_CMO, REF_CMO],
    "cs": {"crs": [
        {
            "name": "horizontal",
            "axes": {
                "xi_rho": {"abbreviation": "X", "coordinates": [{"direction": "east", "unit": "degrees", "values": {"regular": [-70.0, 10.0/N_XI]}}]},
                "eta_rho": {"abbreviation": "Y", "coordinates": [{"direction": "north", "unit": "degrees", "values": {"regular": [30.0, 8.0/N_ETA]}}]}
            }
        },
        {
            "name": "ocean s-coordinate",
            "axes": {
                "s_rho": {
                    "abbreviation": "Z",
                    "coordinates": [{
                        "direction": "up", "unit": "1",
                        "values": {"external": {"ref": {"node": "../params/s_rho"}}},
                        "parametric": {
                            "formula": "ocean_s_coordinate_g2",
                            "terms": {
                                "s":  {"external": {"ref": {"node": "../params/s_rho"}}},
                                "C":  {"external": {"ref": {"node": "../params/Cs_r"}}},
                                "h":  {"external": {"ref": {"node": "../params/h"}}},
                                "hc": {"external": {"ref": {"node": "../params/hc"}}}
                            }
                        }
                    }]
                }
            }
        },
        {
            "name": "temporal",
            "axes": {
                "time": {
                    "abbreviation": "T",
                    "coordinates": [{
                        "direction": "future",
                        "time": {"unit": "days", "epoch": "1850-01-01", "calendar": "standard"},
                        "values": {"external": {"ref": {"node": "../coords/time"}}}
                    }]
                }
            }
        }
    ]}
})

print(f"Store written to {STORE_PATH}")

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
