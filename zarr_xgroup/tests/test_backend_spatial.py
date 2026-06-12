"""
Tests for the spatial convention handler.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from zarr_xgroup.errors import XGroupNoPrincipalWarning

STORE = Path(__file__).parent / "stores" / "spatial_test.zarr"


@pytest.fixture(scope="module")
def spatial_store():
    if not STORE.exists():
        pytest.skip(f"spatial test store not found at {STORE}")
    return STORE


# ---------------------------------------------------------------------------
# /single — all spatial: attrs at array level
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_single(spatial_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return xr.open_dataset(spatial_store, engine="xgroup", group="/single")


def test_single_no_principal_warning(spatial_store):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xr.open_dataset(spatial_store, engine="xgroup", group="/single")
    no_principal = [x for x in w if issubclass(x.category, XGroupNoPrincipalWarning)]
    assert len(no_principal) == 0


def test_single_data_var(ds_single):
    assert "band" in ds_single.data_vars


def test_single_x_coord(ds_single):
    assert "x" in ds_single.coords
    x = ds_single.coords["x"].values
    assert x.shape == (100,)
    # pixel registration: first centre = 300000 + 0.5*10 = 300005
    assert x[0] == pytest.approx(300005.0, abs=1e-3)
    assert x[1] - x[0] == pytest.approx(10.0, abs=1e-6)


def test_single_y_coord(ds_single):
    assert "y" in ds_single.coords
    y = ds_single.coords["y"].values
    assert y.shape == (100,)
    # pixel registration, north-up: first centre = 5200000 + 0.5*(-10) = 5199995
    assert y[0] == pytest.approx(5199995.0, abs=1e-3)
    assert y[1] - y[0] == pytest.approx(-10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# /shared — group-level spatial: attrs inherited by arrays
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_shared(spatial_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return xr.open_dataset(spatial_store, engine="xgroup", group="/shared")


def test_shared_no_principal_warning(spatial_store):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xr.open_dataset(spatial_store, engine="xgroup", group="/shared")
    no_principal = [x for x in w if issubclass(x.category, XGroupNoPrincipalWarning)]
    assert len(no_principal) == 0


def test_shared_both_bands(ds_shared):
    assert "b05" in ds_shared.data_vars
    assert "b06" in ds_shared.data_vars


def test_shared_x_inherited(ds_shared):
    assert "x" in ds_shared.coords
    x = ds_shared.coords["x"].values
    assert x.shape == (50,)
    # 20m pixels: first centre = 300000 + 0.5*20 = 300010
    assert x[0] == pytest.approx(300010.0, abs=1e-3)


def test_shared_y_inherited(ds_shared):
    assert "y" in ds_shared.coords
    y = ds_shared.coords["y"].values
    assert y.shape == (50,)
    # 20m pixels north-up: first centre = 5200000 + 0.5*(-20) = 5199990
    assert y[0] == pytest.approx(5199990.0, abs=1e-3)


# ---------------------------------------------------------------------------
# /node_reg — node registration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_node(spatial_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return xr.open_dataset(spatial_store, engine="xgroup", group="/node_reg")


def test_node_reg_x_no_offset(ds_node):
    """Node registration: first coordinate equals c exactly (no 0.5 offset)."""
    assert "x" in ds_node.coords
    x = ds_node.coords["x"].values
    # transform[2] (c) = -10.0, no offset for node registration
    assert x[0] == pytest.approx(-10.0, abs=1e-9)


def test_node_reg_y_no_offset(ds_node):
    assert "y" in ds_node.coords
    y = ds_node.coords["y"].values
    # transform[5] (f) = 50.0
    assert y[0] == pytest.approx(50.0, abs=1e-9)
