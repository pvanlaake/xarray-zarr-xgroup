"""
Test group= argument behaviour with the xgroup backend against the EOPF store.
"""
import warnings
import xarray as xr
from zarr_xgroup.errors import XGroupNoPrincipalWarning
import ssl
import certifi
ssl._create_default_https_context = ssl.create_default_context
import os
os.environ["SSL_CERT_FILE"] = certifi.where()

S2_URL = (
    "https://objects.eodc.eu/"
    "e05ab01a9d56408d82ac32d69a5aae2a:sample-data/tutorial_data/cpm_v253/"
    "S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458.zarr"
)
OPTS = {"ssl": False}

print("=== open_dataset: single group /measurements/reflectance/r10m ===")
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    ds = xr.open_dataset(S2_URL, engine="xgroup", group="/measurements/reflectance/r10m",
                         storage_options=OPTS)
print(ds)

print()
print("=== open_datatree: rooted at /measurements ===")
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    dt = xr.open_datatree(S2_URL, engine="xgroup", group="/measurements",
                          storage_options=OPTS)
print(dt)
print(f"Groups: {list(dt.groups)}")
