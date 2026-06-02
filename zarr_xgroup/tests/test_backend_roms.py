"""
End-to-end tests of XGroupBackendEntrypoint against the synthetic ROMS
test store.

The ROMS store uses plain CF structural attributes rather than the cs
convention, so XGroupNoPrincipalWarning is expected for every array.
"""
import warnings
from pathlib import Path

import pytest
import xarray as xr

from zarr_xgroup.errors import XGroupNoPrincipalWarning

# Locate the store relative to this test file
STORE = Path(__file__).parent / "stores" / "roms_test.zarr"


@pytest.fixture(scope="module")
def roms_store():
    if not STORE.exists():
        pytest.skip(f"ROMS test store not found at {STORE}")
    return STORE


def test_open_dataset_ocean_has_data_vars(roms_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ds = xr.open_dataset(roms_store, engine="xgroup", group="/ocean")
    assert "temp" in ds.data_vars
    assert "u" in ds.data_vars
    assert "zeta" in ds.data_vars


def test_open_dataset_ocean_emits_no_principal_warning(roms_store):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xr.open_dataset(roms_store, engine="xgroup", group="/ocean")
    no_principal = [x for x in w if issubclass(x.category, XGroupNoPrincipalWarning)]
    assert len(no_principal) > 0, "Expected XGroupNoPrincipalWarning for plain CF store"


def test_open_dataset_ocean_dims(roms_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ds = xr.open_dataset(roms_store, engine="xgroup", group="/ocean")
    assert "time" in ds.dims
    assert "s_rho" in ds.dims
    assert "eta_rho" in ds.dims
    assert "xi_rho" in ds.dims


def test_open_dataset_grid(roms_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ds = xr.open_dataset(roms_store, engine="xgroup", group="/grid")
    assert "lon_rho" in ds.data_vars
    assert "lat_rho" in ds.data_vars
    assert "h" in ds.data_vars


def test_open_datatree_nodes(roms_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        dt = xr.open_datatree(roms_store, engine="xgroup")
    groups = list(dt.groups)
    assert "/" in groups
    assert "/grid" in groups
    assert "/ocean" in groups
    assert "/time" in groups


def test_open_datatree_ocean_node(roms_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        dt = xr.open_datatree(roms_store, engine="xgroup")
    ocean = dt["/ocean"]
    assert "temp" in ocean.data_vars
    assert "u" in ocean.data_vars


def test_open_datatree_grid_node(roms_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        dt = xr.open_datatree(roms_store, engine="xgroup")
    grid = dt["/grid"]
    assert "lon_rho" in grid.data_vars
    assert "h" in grid.data_vars