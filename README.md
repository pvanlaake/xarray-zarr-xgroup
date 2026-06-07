# xarray-zarr-xgroup

An XArray backend that correctly implements the Zarr data model, including full
group hierarchy and cross-group reference resolution.

## The problem

XArray's default Zarr backend opens one group at a time and silently drops any
coordinate or ancillary references that point outside that group. For complex,
hierarchical data products — such as ESA's EOPF Sentinel format or ROMS ocean
model output — this means coordinates are missing, variables are uninterpretable,
and the only workarounds are data duplication or lossy flattening of the hierarchy.

## The solution

`xarray-zarr-xgroup` resolves cross-group references before handing data to
XArray, so that `Dataset` and `DataTree` objects are fully populated regardless
of where referenced arrays live in the store.

```python
import xarray as xr

# All cross-group coordinate references resolved automatically
ds = xr.open_dataset("my_store.zarr", engine="xgroup", group="/ocean")

# Full hierarchy with resolved references at every node
dt = xr.open_datatree("my_store.zarr", engine="xgroup")
```

## Quick start

### Opening a single group

```python
import xarray as xr

ds = xr.open_dataset(
    "my_store.zarr",
    engine="xgroup",
    group="/measurements/reflectance/r10m",  # optional, defaults to root
)
print(ds)
```

### Opening the full hierarchy as a DataTree

```python
import xarray as xr

dt = xr.open_datatree("my_store.zarr", engine="xgroup")
print(dt)
print(list(dt.groups))
```

### Opening a remote store

```python
import xarray as xr

S2_URL = (
    "https://objects.eodc.eu/"
    "e05ab01a9d56408d82ac32d69a5aae2a:sample-data/tutorial_data/cpm_v253/"
    "S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458.zarr"
)

dt = xr.open_datatree(S2_URL, engine="xgroup", storage_options={"ssl": False})
print(dt)
```

Compare with the default XArray Zarr backend on the same store:

```python
ds_default = xr.open_zarr(S2_URL, consolidated=False, storage_options={"ssl": False})
print(ds_default)  # empty Dataset — coordinates and hierarchy both missing
```

### Suppressing or promoting warnings

Arrays without a declared principal convention emit `XGroupNoPrincipalWarning`
and are loaded without coordinate resolution. This is the correct behaviour for
stores that use plain CF attributes rather than a declared GeoZarr convention.

```python
import warnings
from zarr_xgroup.errors import XGroupNoPrincipalWarning

# Suppress (e.g. when opening a plain CF store deliberately)
warnings.filterwarnings("ignore", category=XGroupNoPrincipalWarning)

# Promote to error (e.g. when a cs-annotated store is expected)
warnings.filterwarnings("error", category=XGroupNoPrincipalWarning)
```

## How it works

Interpretation of the coordinate structure of each array is delegated to the
Zarr convention declared in that array's metadata. The backend detects the
active conventions, resolves all cross-group references they declare, and
assembles a fully-resolved variable dict before XArray's own machinery runs.

Two tiers of conventions are supported:

- **Principal conventions** define the coordinate structure of an array.
  Exactly one must be declared per array. Built-in: `cs`, `spatial`.
- **Service conventions** provide auxiliary capabilities that compose
  freely with any principal convention.
  Built-in: `ref`, `proj`, `uom`, `geolocation`.

Additional conventions can be registered via Python entry points:

```toml
[project.entry-points."zarr_xgroup.conventions"]
my_convention = "my_package.convention:MyConventionHandler"
```

## Installation

```bash
pip install xarray-zarr-xgroup
```

Requires Python ≥ 3.11, XArray ≥ 2024.10.0, zarr ≥ 3.0.0.

## Status

Early development. The initial release targets read-only access to Zarr v2
and v3 stores with `cs` and `spatial` principal conventions. See the
[specification](xarray_zarr_xgroup_spec.md) for full details.

## Background

This package grew out of a broader effort to bring correct GeoZarr convention
support to hierarchical Zarr stores and into XArray. For context on the problem and design
decisions, see the [specification](xarray_zarr_xgroup_spec.md).

## License

Apache 2.0