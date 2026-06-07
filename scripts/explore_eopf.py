"""
Open a real-world EOPF Sentinel-2 Zarr store with xarray-zarr-xgroup
and compare with the default XArray Zarr backend.

Store: Sentinel-2 L1C product from the EOPF Sample Service
URL: https://objects.eodc.eu/...
"""
import warnings
from zarr_xgroup.errors import XGroupNoPrincipalWarning
import xarray as xr
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

print("=== Opening with xgroup backend (DataTree) ===")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        dt = xr.open_datatree(S2_URL, engine="xgroup")
        print(dt)
        print(f"\nGroups: {list(dt.groups)[:20]}")
        no_principal = [x for x in w if issubclass(x.category, XGroupNoPrincipalWarning)]
        print(f"\nXGroupNoPrincipalWarning count: {len(no_principal)}")
    except Exception as e:
        print(f"ERROR: {e}")

print()
print("=== Opening root with default zarr backend ===")
try:
    ds_default = xr.open_zarr(S2_URL, consolidated=False)
    print(ds_default)
except Exception as e:
    print(f"ERROR: {e}")
