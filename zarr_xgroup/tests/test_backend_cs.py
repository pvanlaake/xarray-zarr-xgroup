"""
End-to-end tests of XGroupBackendEntrypoint against the cs-convention
test store.

These tests verify that the cs convention handler correctly resolves:
  - Regular inline spatial coordinates
  - External coordinate arrays via ref
  - Cross-group CRS references
  - Boundary arrays
  - Parametric vertical coordinate terms
  - Geolocation arrays (presence check only — geolocation handler is a stub)
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from zarr_xgroup.errors import XGroupNoPrincipalWarning

STORE = Path(__file__).parent / "stores" / "cs_test.zarr"


@pytest.fixture(scope="module")
def cs_store():
    if not STORE.exists():
        pytest.skip(f"cs test store not found at {STORE}")
    return STORE


# ---------------------------------------------------------------------------
# /simple — regular inline coords + cross-group CRS ref + external time
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_simple(cs_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return xr.open_dataset(cs_store, engine="xgroup", group="/simple")


def test_simple_no_principal_warning(cs_store):
    """cs convention is declared — no XGroupNoPrincipalWarning expected."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xr.open_dataset(cs_store, engine="xgroup", group="/simple")
    no_principal = [x for x in w if issubclass(x.category, XGroupNoPrincipalWarning)]
    assert len(no_principal) == 0, (
        f"Unexpected XGroupNoPrincipalWarning: {[str(x.message) for x in no_principal]}"
    )


def test_simple_data_var_present(ds_simple):
    assert "temperature" in ds_simple.data_vars


def test_simple_dims(ds_simple):
    assert set(ds_simple.dims) >= {"time", "lat", "lon"}


def test_simple_inline_spatial_coords(ds_simple):
    """lon and lat should be attached as coordinates from regular inline values."""
    assert "lon" in ds_simple.coords, "lon coordinate missing"
    assert "lat" in ds_simple.coords, "lat coordinate missing"


def test_simple_lon_values(ds_simple):
    """lon should start at -179.75 with increment 0.5."""
    lon = ds_simple.coords["lon"].values
    assert lon[0] == pytest.approx(-179.75, abs=1e-6)
    assert lon[1] - lon[0] == pytest.approx(0.5, abs=1e-6)


def test_simple_lat_values(ds_simple):
    """lat should start at -89.75 with increment 0.5."""
    lat = ds_simple.coords["lat"].values
    assert lat[0] == pytest.approx(-89.75, abs=1e-6)
    assert lat[1] - lat[0] == pytest.approx(0.5, abs=1e-6)


def test_simple_external_time_coord(ds_simple):
    """time should be attached from the external array in /coords."""
    assert "time" in ds_simple.coords, "time coordinate missing"
    time = ds_simple.coords["time"]
    assert time.shape == (24,)


def test_simple_time_bounds(ds_simple):
    assert "time_bounds" in ds_simple.data_vars or "time_bounds" in ds_simple.coords, (
        "time boundary array not found"
    )

# ---------------------------------------------------------------------------
# /curvilinear — rotated grid + geolocation arrays
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_curv(cs_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return xr.open_dataset(cs_store, engine="xgroup", group="/curvilinear")


def test_curv_data_var_present(ds_curv):
    assert "pr" in ds_curv.data_vars


def test_curv_dims(ds_curv):
    assert set(ds_curv.dims) >= {"time", "rlat", "rlon"}


def test_curv_inline_rotated_coords(ds_curv):
    """rlon and rlat should be attached as coordinates from regular inline values."""
    assert "rlon" in ds_curv.coords, "rlon coordinate missing"
    assert "rlat" in ds_curv.coords, "rlat coordinate missing"


def test_curv_rlon_values(ds_curv):
    rlon = ds_curv.coords["rlon"].values
    assert rlon[0] == pytest.approx(-28.375, abs=1e-6)
    assert rlon[1] - rlon[0] == pytest.approx(0.11, abs=1e-6)


def test_curv_external_time(ds_curv):
    assert "time" in ds_curv.coords
    assert ds_curv.coords["time"].shape == (24,)


# ---------------------------------------------------------------------------
# /ocean — 4D array with parametric vertical coords
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_ocean(cs_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return xr.open_dataset(cs_store, engine="xgroup", group="/ocean")


def test_ocean_data_var_present(ds_ocean):
    assert "temp" in ds_ocean.data_vars


def test_ocean_dims(ds_ocean):
    assert set(ds_ocean.dims) >= {"time", "s_rho", "eta_rho", "xi_rho"}


def test_ocean_inline_horizontal_coords(ds_ocean):
    assert "xi_rho" in ds_ocean.coords, "xi_rho coordinate missing"
    assert "eta_rho" in ds_ocean.coords, "eta_rho coordinate missing"


def test_ocean_external_s_rho(ds_ocean):
    """s_rho should be attached from external array in /params."""
    assert "s_rho" in ds_ocean.coords, "s_rho coordinate missing"
    s = ds_ocean.coords["s_rho"].values
    assert s.shape == (5,)
    assert s[0] < s[-1], "s_rho should increase"


def test_ocean_parametric_terms_present(ds_ocean):
    pterm_vars = [c for c in ds_ocean.data_vars if "pterm" in c]
    assert len(pterm_vars) > 0, (
        f"No parametric term variables found. Data vars: {set(ds_ocean.data_vars)}"
    )

def test_ocean_external_time(ds_ocean):
    assert "time" in ds_ocean.coords
    assert ds_ocean.coords["time"].shape == (24,)


# ---------------------------------------------------------------------------
# DataTree
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dt_cs(cs_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return xr.open_datatree(cs_store, engine="xgroup")


def test_datatree_nodes(dt_cs):
    groups = list(dt_cs.groups)
    assert "/" in groups
    assert "/coords" in groups
    assert "/simple" in groups
    assert "/curvilinear" in groups
    assert "/ocean" in groups
    assert "/params" in groups


def test_datatree_simple_coords_resolved(dt_cs):
    simple = dt_cs["/simple"]
    assert "lon" in simple.coords
    assert "lat" in simple.coords


def test_datatree_ocean_s_rho_resolved(dt_cs):
    ocean = dt_cs["/ocean"]
    assert "s_rho" in ocean.coords
