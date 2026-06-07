"""
Generate a minimal synthetic ROMS-like Zarr v3 store.

CF-correct but requires cross-group reference resolution to be fully
interpreted. Every CF reference (coordinates, ancillary_variables,
formula_terms) crosses at least one group boundary — this is intentional,
to test backends that implement full cross-group resolution.

Group layout:
  /                     root (global attributes only)
  /grid/                all coordinate and ancillary arrays
  /ocean/               data variables
  /time/                time coordinate (separate group, deliberately)

Usage:
    python make_roms_test_store.py

Output: zarr_xgroup/tests/stores/roms_test.zarr
"""

import numpy as np
import zarr
import os, shutil
from pathlib import Path

STORE_PATH = Path(__file__).parent.parent / "zarr_xgroup" / "tests" / "stores" / "roms_test.zarr"

N_TIME    = 3
N_S_RHO   = 5
N_ETA_RHO = 8
N_XI_RHO  = 10
N_ETA_U   = N_ETA_RHO
N_XI_U    = N_XI_RHO - 1

def ca(grp, name, data, dims, chunks=None, **attrs):
    """Create array with dimension_names and CF attributes."""
    if chunks is None:
        chunks = data.shape if data.ndim > 0 else ()
    arr = grp.create_array(
        name,
        data=data,
        chunks=chunks if chunks else (),
        dimension_names=list(dims),
    )
    arr.attrs.update(attrs)
    return arr

if os.path.exists(STORE_PATH):
    shutil.rmtree(STORE_PATH)
STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

root = zarr.open(str(STORE_PATH), mode="w")
root.attrs.update({
    "Conventions": "CF-1.11",
    "title": "Synthetic minimal ROMS-like Zarr v3 test store",
    "institution": "cf-zarr test fixture",
    "source": "synthetic",
    "comment": (
        "All CF cross-references span group boundaries intentionally, "
        "to exercise backends that implement full cross-group resolution."
    ),
})

rng = np.random.default_rng(42)

# ---- /time -----------------------------------------------------------------
gt = root.require_group("time")

ca(gt, "ocean_time",
   data=np.array([0.0, 86400.0, 172800.0], dtype="f8"),
   dims=["time"],
   standard_name="time",
   long_name="ocean time",
   units="seconds since 2000-01-01 00:00:00",
   calendar="gregorian",
   axis="T",
)

# ---- /grid -----------------------------------------------------------------
gg = root.require_group("grid")

xi  = np.linspace(-70.0, -60.0, N_XI_RHO)
eta = np.linspace( 30.0,  38.0, N_ETA_RHO)
XI, ETA = np.meshgrid(xi, eta)
lon_rho_d = XI  + 0.05 * np.sin(np.pi * ETA / 40.0)
lat_rho_d = ETA + 0.05 * np.cos(np.pi * XI  / 70.0)

ca(gg, "lon_rho", lon_rho_d.astype("f8"), ["eta_rho", "xi_rho"],
   standard_name="longitude", long_name="longitude of rho-points", units="degrees_east")
ca(gg, "lat_rho", lat_rho_d.astype("f8"), ["eta_rho", "xi_rho"],
   standard_name="latitude",  long_name="latitude of rho-points",  units="degrees_north")

xi_u = 0.5 * (xi[:-1] + xi[1:])
XI_U, ETA_U = np.meshgrid(xi_u, eta)
lon_u_d = XI_U  + 0.05 * np.sin(np.pi * ETA_U / 40.0)
lat_u_d = ETA_U + 0.05 * np.cos(np.pi * XI_U  / 70.0)

ca(gg, "lon_u", lon_u_d.astype("f8"), ["eta_u", "xi_u"],
   standard_name="longitude", long_name="longitude of u-points", units="degrees_east")
ca(gg, "lat_u", lat_u_d.astype("f8"), ["eta_u", "xi_u"],
   standard_name="latitude",  long_name="latitude of u-points",  units="degrees_north")

h_d = 200.0 + 800.0 * rng.random((N_ETA_RHO, N_XI_RHO))
ca(gg, "h", h_d.astype("f8"), ["eta_rho", "xi_rho"],
   standard_name="sea_floor_depth_below_geoid",
   long_name="bathymetry at rho-points", units="m", positive="down")

ca(gg, "hc", np.array(20.0, dtype="f8"), dims=[],
   long_name="S-coordinate parameter, critical depth", units="m")

s_rho_d = np.linspace(-1.0, 0.0, N_S_RHO, endpoint=False) + 0.5 / N_S_RHO
ca(gg, "s_rho", s_rho_d.astype("f8"), ["s_rho"],
   standard_name="ocean_s_coordinate_g2",
   long_name="S-coordinate at rho-points",
   units="1", positive="up", axis="Z",
   formula_terms=(
       "s: /grid/s_rho  C: /grid/Cs_r  eta: /ocean/zeta  "
       "h: /grid/h  hc: /grid/hc"
   ),
)

theta_s, theta_b = 6.0, 0.9
Cs_r_d = (
    (1 - theta_b) * np.sinh(theta_s * s_rho_d) / np.sinh(theta_s)
    + theta_b * (
        np.tanh(theta_s * (s_rho_d + 0.5)) / (2 * np.tanh(0.5 * theta_s)) - 0.5
    )
)
ca(gg, "Cs_r", Cs_r_d.astype("f8"), ["s_rho"],
   long_name="S-coordinate stretching curves at rho-points",
   units="1", valid_min=-1.0, valid_max=0.0)

mask_d = (rng.random((N_ETA_RHO, N_XI_RHO)) > 0.15).astype("f4")
ca(gg, "mask_rho", mask_d, ["eta_rho", "xi_rho"],
   long_name="land/sea mask at rho-points",
   flag_values=[0.0, 1.0], flag_meanings="land water")

# ---- /ocean ----------------------------------------------------------------
go = root.require_group("ocean")

zeta_d = 0.1 * rng.standard_normal((N_TIME, N_ETA_RHO, N_XI_RHO))
ca(go, "zeta", zeta_d.astype("f4"), ["time", "eta_rho", "xi_rho"],
   chunks=(1, N_ETA_RHO, N_XI_RHO),
   standard_name="sea_surface_height_above_geoid",
   long_name="free-surface", units="m",
   coordinates="/time/ocean_time /grid/lon_rho /grid/lat_rho",
   ancillary_variables="/grid/mask_rho",
)

temp_d = (
    15.0
    + 5.0 * s_rho_d[np.newaxis, :, np.newaxis, np.newaxis]
    + 0.5 * rng.standard_normal((N_TIME, N_S_RHO, N_ETA_RHO, N_XI_RHO))
)
ca(go, "temp", temp_d.astype("f4"),
   ["time", "s_rho", "eta_rho", "xi_rho"],
   chunks=(1, N_S_RHO, N_ETA_RHO, N_XI_RHO),
   standard_name="sea_water_potential_temperature",
   long_name="potential temperature", units="degrees_C",
   coordinates="/time/ocean_time /grid/s_rho /grid/lon_rho /grid/lat_rho",
   ancillary_variables="/grid/mask_rho",
)

u_d = 0.1 * rng.standard_normal((N_TIME, N_S_RHO, N_ETA_U, N_XI_U))
ca(go, "u", u_d.astype("f4"),
   ["time", "s_rho", "eta_u", "xi_u"],
   chunks=(1, N_S_RHO, N_ETA_U, N_XI_U),
   standard_name="sea_water_x_velocity",
   long_name="u-momentum component", units="m s-1",
   coordinates="/time/ocean_time /grid/s_rho /grid/lon_u /grid/lat_u",
)

# ---------------------------------------------------------------------------
def inventory(grp, indent=0):
    pad = "  " * indent
    for name in sorted(grp.array_keys()):
        arr = grp[name]
        dims = arr.metadata.dimension_names
        print(f"{pad}  [{name}]  shape={arr.shape}  dims={dims}")
        for key in ("coordinates", "ancillary_variables", "formula_terms"):
            if key in arr.attrs:
                print(f"{pad}      {key}: {arr.attrs[key]}")
    for name in sorted(grp.group_keys()):
        print(f"{pad}  /{name}/")
        inventory(grp[name], indent + 1)

print(f"Store written to {STORE_PATH}\n")
print("=== Inventory ===")
inventory(root)
