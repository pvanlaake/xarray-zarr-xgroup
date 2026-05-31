# xarray-zarr-xgroup

An XArray backend that opens a Zarr store, including full
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

## How it works

Interpretation of the coordinate structure of each array is delegated to the
Zarr convention declared in that array's metadata. The backend detects the
active conventions, collects all cross-group references they declare, resolves
the reference paths throughout the store, and assembles a fully-resolved
variable dict before XArray's own decoding machinery runs.

Two tiers of conventions are supported:

- **Principal conventions** define the coordinate structure of an array.
  Exactly one must be declared per array. Built-in: `cs`, `spatial`.
- **Service conventions** provide auxiliary capabilities (CRS definitions,
  cross-store references, units of measure, geolocation arrays) that compose
  freely with any principal convention. Built-in: `ref`, `proj`, `uom`,
  `geolocation`.

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
and v3 stores with `cs` and `spatial` principal conventions. Write support
and convention-specific semantic transformations (formula evaluation,
reprojection) are out of scope for the initial release.

## Background

This package grew out of a broader effort to bring correct CF and GeoZarr
convention support to hierarchical Zarr stores. For context on the problem
and the design decisions, see the
[specification](xarray_zarr_xgroup_spec.md).

## License

Apache 2.0