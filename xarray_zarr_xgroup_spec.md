# xarray-zarr-xgroup Specification

**Package name:** `xarray-zarr-xgroup`
**Import name:** `zarr_xgroup`
**Engine name:** `xgroup`

**Version:** initial release (draft)
**Date:** 2026-05-31
**Status:** Pre-implementation

---

## 1. Motivation

### 1.1 XArray

XArray is a very widely used Python module for the analysis of multi-dimensional array data. It was developed to import netCDF data formatted using the CF Metadata Conventions into Python. Being based on the ["classic" netCDF-3 data model](https://docs.unidata.ucar.edu/netcdf-c/4.9.3/netcdf_data_model.html) data sets are logically contained in a single group, including all ancillary data such as coordinates and attributes. XArray interprets all the information it finds in the group and produces a `Dataset` instance, with one or more variables, and coordinate values taken from "coordinate variables".

After XArray was initially released the hierarchical data format has become more common, for instance HDF5 which is being used in the Common Data Model underlying the newer netCDF-4 format. Groups can be nested in hierarchies, enabling a more expressive, logical and efficient data storage format. Individual arrays can be referenced and used by any other array through within-file path traversal.

XArray has only partial support for such hierarchical stores. Hierarchies may be discovered through the construction of a `DataTree`, or a `Dataset` can be constructed anywhere in the hierarchy by using the `groups =` argument to `open_dataset()`. `Dataset` instances are still self-contained within a single group, though, and a `DataTree` is a collection of such instances encountered throughout the data store. Out-of-group references are silently (!) dropped, leading to the dreaded "Dimensions without coordinates:" list.

Given XArray's wide user base, data producers and processors employ a variety of strategies to make their data products compatible with XArray. Data producers avoid out-of-group references, which effectively results in a handicapped hierarchy and a host of other problems such as (coordinate) data duplication. Data processors may flatten true hierarchical data stores to the single-group model of XArray, turning full path references into composite names in the flat group, a processing overhead that is error-prone.

### 1.2 Zarr

Zarr is a relative newcomer in the multi-dimensional array world. It is hierarchical by design ("Hierarchy" is the first concept that is defined in the specification) and the core specification requires implementations to support path discovery and traversal. While the specification is silent on the issue of cross-referencing between separate arrays (it is deliberately agnostic of any specific application built on Zarr), the format is well-suited to support it, such as through its use of `path` and `prefix` arguments in core API functions that are typically relative to the root of the Zarr store.

Zarr conventions are community conventions on the structure and semantics of the arrays in the Zarr store. Typical examples of these are coordinates mapped onto the Zarr array indexing scheme, standard referencing of out-of-group objects and attributes, licensing, units-of-measure, etc. GeoZarr is a community effort to build conventions to describe geospatial data. In the context of this specification we can identify two broad groups of conventions:

1. **Principal conventions** provide coordinates for all of the axes that form the coordinate system of the array. Two principal conventions will be supported by this project initially. The `cs` (coordinate set) convention is a comprehensive scheme to attach coordinates to any axis of the array. It is based on the OGC standard "Referencing by Coordinates" and implements constructs from the CF Metadata Conventions. It can describe simple coordinate systems but also complex arrangements with multiple coordinate sets per axis, coordinate boundary values, model calendars for the temporal domain, parametric vertical coordinates and curvilinear geolocation arrays, as needed. The `spatial` convention is much more compact and focused on the spatial axes of imagery-type data sets, such as satellite images. It aligns very well with other geospatial tools in widespread use.
2. **Service conventions** provide additional structure or information to the principal convention. The `proj` convention, for example, identifies the coordinate reference system (how spatial coordinates map to the Earth) of the coordinate system. The `ref` convention defines a standard way of referencing a node (group, array) or its attributes from the current node. The initially supported service conventions of this project are those which are referenced by the `cs` and `spatial` principal conventions.

### 1.3 Breaking out of the group

XArray can read and write Zarr stores but it does so with its default approach on containment. A `Dataset` instance written by XArray to a Zarr store is located in a single group and has the tell-tale signs from its netCDF-CF heritage, such as the `coordinates` attribute.

There is a large range of data products where a single, logical scientific data set is composed of multiple arrays. The number of arrays in a single data set can easily reach several dozens, such as with ESA's new EOPF format for Sentinel data and ROMS data in the ocean modeling community. Placing all of those arrays in a single group is usually not a viable solution due to technical reasons or simply because the arrays may have a natural grouping, such as by resolution in the EOPF products, or the vertical parameter arrays in ROMS.

Getting these complex data products into XArray is a valuable proposition, due to its large user base and the availability of tools for further processing and analysis available in Python. It is essential, though, to break through the artificial group boundary that XArray imposes on itself to fully capitalize on the features of a hierarchical data store like Zarr. Fortunately, XArray has opened that door by allowing third-party backends to register themselves to make a format available to XArray as a `Dataset` or `DataTree`.

**`xarray-zarr-xgroup`** is an XArray backend that processes Zarr stores, including full group hierarchy and cross-group reference resolution. It produces `Dataset` and `DataTree` objects in which all secondary nodes referenced by arrays in the opened store or group are present and correctly attached for XArray to use. Interpretation of the declared structure of the group or array is delegated to the Zarr conventions referenced in the metadata of the array or group. The conventions themselves are modular and additional conventions may be registered with this backend.

---

## 2. Principal properties

1. Backend for XArray, registered as a `BackendEntrypoint` under engine name `xgroup`.
2. Must ingest Zarr v2 and v3 stores.
3. Must support full cross-group reference resolution throughout the store, and across stores where feasible.
4. Must support a two-tier composable convention handler model and be extensible to future conventions via a Python entry point registry.
5. Must support both `Dataset` and `DataTree` output.

---

## 3. Scope

### 3.1 Initial release

- Opening a Zarr store or any group therein
- Full traversal of the group hierarchy
- Resolution of cross-group structural references to secondary nodes as per the declared convention of the array
- Cross-store reference resolution for publicly accessible stores reachable via standard zarr-python storage backends (local filesystem, S3, GCS, HTTP)
- Lazy loading throughout; reference resolution is a metadata-only operation
- `Dataset` output: single group with all referenced secondary nodes resolved into the variable dict
- `DataTree` output: full hierarchy with cross-group references resolved between nodes
- Read-only access
- Zarr v2 and v3 format support
- Built-in principal conventions: `cs`, `spatial`
- Built-in service conventions: `ref`, `proj`, `uom`, `geolocation`
- Arrays without a declared principal convention are loaded as plain data variables without secondary node resolution; a warning is emitted

### 3.2 Out of scope for the initial release

- Write support
- Any convention-specific semantic transformation (formula evaluation, CRS reprojection, resampling, unit conversion)
- Authenticated cross-store references

---

## 4. Architecture

### 4.1 Entry point

`xarray-zarr-xgroup` registers `XGroupBackendEntrypoint` subclassing XArray's `BackendEntrypoint`. It declares `supports_groups = True`.

`guess_can_open()` returns `True` for any path or store that zarr-python can open. Users invoke the backend explicitly via `engine='xgroup'`.

### 4.2 Processing pipeline

```
`XGroupBackendEntrypoint.open_dataset()` / `open_datatree()`
    │
    ├── 1. Store opening
    │       `zarr.open(store_path, mode='r')`
    │       Full root store access regardless of target group
    │
    ├── 2. Hierarchy traversal
    │       Walk group tree from target group
    │       Collect all arrays and their metadata
    │
    ├── 3. Reference resolution (per array)
    │       Detect active principal convention handler
    │           → if none: emit `XGroupNoPrincipalWarning`, skip resolution
    │       Detect active service convention handlers (zero or more)
    │       Collect all Reference objects from active convention handlers
    │       Resolve each reference path
    │       Fetch referenced arrays as lazy `ZarrArrayWrapper` objects
    │       Add to resolved variable dict
    │
    ├── 4. Variable dict assembly
    │       Construct `ResolvedZarrStore` implementing `AbstractDataStore`
    │       Rewrite reference path strings to flat resolved names
    │       `get_variables()` returns fully resolved `FrozenDict`
    │       `get_attrs()` returns group attributes
    │       `get_dimensions()` derived from `"dimension_names"`" / `"_ARRAY_DIMENSIONS"`"
    │
    └── 5. XArray handoff
            `StoreBackendEntrypoint.open_dataset(ResolvedZarrStore)
            → Dataset`

            For `DataTree`: repeat per group, assemble via `DataTree.from_dict()`
```

### 4.3 Convention handler model

Conventions are split into two tiers that compose freely. Detection and resolution operate at the individual array level, consistent with Zarr's philosophy that every array is self-describing and can stand on its own.

#### Principal conventions

A principal convention defines the coordinate structure of an array — how its axes are described and how secondary coordinate nodes are referenced. Exactly one principal convention must be declared per array.

Built-in principal conventions (v1):

| Name | Purpose |
|---|---|
| `cs` | Coordinate set convention; explicit coordinate structure declaration |
| `spatial` | Spatial coordinate convention for imagery-type data; aligned with common geospatial tooling |

#### Service conventions

A service convention provides an auxiliary capability orthogonal to coordinate structure. Multiple service conventions may be active simultaneously. The `geolocation` service convention is always used in combination with a principal convention (`cs` or `spatial`), never standalone.

Built-in service conventions (v1):

| Name | Purpose |
|---|---|
| `ref` | Cross-node and cross-store references |
| `proj` | CRS description via PROJJSON or WKT2 |
| `uom` | Unit of measure definitions |
| `geolocation` | Geolocation arrays for curvilinear grids; always paired with a principal convention |

#### Handler interface

All handlers implement a common base:

```python
class ConventionHandler:
    tier: Literal["principal", "service"]
    name: str

    @staticmethod
    def detect(root: zarr.Group, group: zarr.Group, array: zarr.Array) -> bool:
        """Return True if this convention applies to this array."""

    def references(self, array: zarr.Array, group: zarr.Group) -> list[Reference]:
        """
        Return all structural references declared by this array under
        this convention.
        """
```

The `Reference` dataclass:

```python
@dataclass
class Reference:
    source_array_path: str    # absolute path of the declaring array
    source_role: str          # role label from the convention
    target_path: str          # path as declared in metadata, before resolution
    resolved_path: str | None # absolute path after resolution; None if failed
    convention: str           # name of the handler that produced this reference
```

#### Composability

For each array the backend:

1. Applies the active principal handler → coordinate-structural references
2. Applies each active service handler → service references
3. Merges all `Reference` lists
4. Resolves all target paths

References from different handlers are independent and non-conflicting: principal handlers own coordinate structure, service handlers own their namespaced attributes.

#### Registry and extensibility

Built-in handlers are registered at package import. Third-party handlers register via Python entry points:

```toml
[project.entry-points."zarr_xgroup.conventions"]
my_convention = "my_package.convention:MyConventionHandler"
```

### 4.4 Path resolution

1. **Absolute paths** beginning with `/` are resolved relative to the root of the current store.
2. **Relative paths** are resolved relative to the group containing the source array using `../` traversal per RFC 3986.
3. **Cross-store URIs** containing `://` are opened as separate zarr stores via zarr-python's storage backend machinery. Storage options are supplied via `cross_store_storage_options`.
4. Path resolution is a pure metadata operation; no chunk data is read.

### 4.5 ResolvedZarrStore

`ResolvedZarrStore` implements XArray's `AbstractDataStore` interface:

- `get_variables()`: returns a `FrozenDict` of all arrays in the target group plus all resolved secondary arrays, each wrapped as a lazy `ZarrArrayWrapper`. Reference path strings in variable attributes are rewritten to flat resolved names so that XArray's coordinate attachment logic finds all names present in the dict.
- `get_attrs()`: returns the target group's attributes.
- `get_dimensions()`: derived from `dimension_names` (v3) or `_ARRAY_DIMENSIONS` (v2) across all arrays in the resolved variable dict.

---

## 5. Output contract

### 5.1 Dataset output

`open_dataset()` returns an `xr.Dataset` in which:

- All arrays physically present in the target group appear as data variables or dimension coordinates
- All secondary arrays resolved from cross-group references are present in the variable dict and correctly attached as coordinates
- Arrays without a declared principal convention appear as plain data variables with their original attributes intact and no coordinates attached
- Dimension names are assigned from `dimension_names` (v3) or `_ARRAY_DIMENSIONS` (v2)
- All `Variable.attrs` reflect the original array attributes
- All `Variable.encoding` reflects zarr storage parameters for round-trip fidelity
- All arrays are lazy; no chunk data has been read

### 5.2 DataTree output

`open_datatree()` returns an `xr.DataTree` in which:

- Each group in the hierarchy becomes a `DataTree` node
- Each node's `Dataset` satisfies the Dataset output contract above
- Cross-group references are resolved into the Dataset of the node that declares them
- The tree root corresponds to the store root or the `group` argument

---

## 6. Error handling and warnings

Arrays without a declared principal convention emit `XGroupNoPrincipalWarning` and are loaded without secondary node resolution. This warning is suppressable via standard Python `warnings` machinery and may be promoted to an error by the caller if strict behaviour is required:

```python
import warnings
from zarr_xgroup.errors import XGroupNoPrincipalWarning

# promote to error
warnings.filterwarnings("error", category=XGroupNoPrincipalWarning)
```

Unresolvable reference paths raise `XGroupReferenceError` identifying the source array path, the offending attribute, the unresolved target path, and the reason for failure.

| Condition | Behaviour |
|---|---|
| No principal convention detected | `XGroupNoPrincipalWarning`; array loaded without resolution |
| Reference target path not found | `XGroupReferenceError` raised |
| Reference target store unreachable | `XGroupStoreError` raised |
| Malformed reference path | `XGroupPathError` raised |

---

## 7. API

```python
import xarray as xr

# Dataset — single group, all cross-group references resolved
ds = xr.open_dataset(
    store,
    engine="xgroup",
    group="/ocean",                        # optional, default root
    cross_store_storage_options=None,      # dict of url_prefix → storage_options
)

# DataTree — full hierarchy, all cross-group references resolved
dt = xr.open_datatree(
    store,
    engine="xgroup",
    cross_store_storage_options=None,
)
```

All standard XArray `open_dataset` parameters pass through to `StoreBackendEntrypoint` unchanged.

---

## 8. Testing contract

### 8.1 Required test stores

| Store | Purpose |
|---|---|
| `roms_test.zarr` | Cross-group reference integration test |
| `flat.zarr` | Regression baseline; single group, no cross-group references |
| `cs_test.zarr` | `cs` principal convention handler |
| `broken_refs.zarr` | Reference error handling verification |
| `no_principal.zarr` | `XGroupNoPrincipalWarning` behaviour verification |

### 8.2 Test separation

Reference resolution logic and convention handlers must be unit-testable independently of the XArray integration. `ResolvedZarrStore` must be constructable from a plain dict without a live Zarr store.

---

## 9. Package structure

```
xarray-zarr-xgroup/
│
├── pyproject.toml
│
└── zarr_xgroup/
    ├── __init__.py
    ├── backend.py          # XGroupBackendEntrypoint, ResolvedZarrStore
    ├── resolver.py         # Reference dataclass, path resolution, traversal
    ├── errors.py           # XGroupError hierarchy, XGroupNoPrincipalWarning
    │
    ├── conventions/
    │   ├── __init__.py     # registry, detection orchestration
    │   ├── base.py         # ConventionHandler base class
    │   ├── cs.py           # cs convention (principal)
    │   ├── spatial.py      # spatial convention (principal)
    │   ├── ref.py          # ref service convention
    │   ├── proj.py         # proj service convention
    │   ├── uom.py          # uom service convention
    │   └── geolocation.py  # geolocation service convention
    │
    └── tests/
        ├── conftest.py
        ├── test_resolver.py
        ├── test_conventions.py
        ├── test_backend.py
        ├── test_datatree.py
        └── stores/
            ├── roms_test.zarr/
            ├── flat.zarr/
            ├── cs_test.zarr/
            ├── broken_refs.zarr/
            └── no_principal.zarr/
```

---

## 10. Dependencies and compatibility

| Dependency | Minimum version |
|---|---|
| Python | 3.11 |
| `xarray` | 2024.10.0 |
| `zarr` | 3.0.0 |
| `numpy` | 1.24.0 |

Version requirement failures raise `ImportError` with explicit guidance at import time.
