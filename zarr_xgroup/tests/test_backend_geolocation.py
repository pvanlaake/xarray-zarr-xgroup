"""
Tests for the geolocation convention handler.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from zarr_xgroup.errors import XGroupNoPrincipalWarning

STORE = Path(__file__).parent / "stores" / "geolocation_test.zarr"


@pytest.fixture(scope="module")
def geo_store():
    if not STORE.exists():
        pytest.skip(f"geolocation test store not found at {STORE}")
    return STORE


# ---------------------------------------------------------------------------
# /data/pr — geodetic geolocation only
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_pr(geo_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return xr.open_dataset(geo_store, engine="xgroup", group="/data")


def test_pr_data_var(ds_pr):
    assert "pr" in ds_pr.data_vars


def test_pr_longitude_resolved(ds_pr):
    """longitude array resolved from ../coords/longitude."""
    assert "longitude" in ds_pr.data_vars or "longitude" in ds_pr.coords, \
        f"longitude not found. vars={list(ds_pr.data_vars)}"


def test_pr_latitude_resolved(ds_pr):
    assert "latitude" in ds_pr.data_vars or "latitude" in ds_pr.coords, \
        f"latitude not found. vars={list(ds_pr.data_vars)}"


def test_pr_geolocation_shape(ds_pr):
    """Geolocation arrays must have same spatial shape as data array."""
    pr_shape = ds_pr["pr"].shape
    if "longitude" in ds_pr.data_vars:
        lon_shape = ds_pr["longitude"].shape
    else:
        lon_shape = ds_pr.coords["longitude"].shape
    assert lon_shape == pr_shape


def test_pr_longitude_values(ds_pr):
    """Longitude values should be in a plausible range."""
    if "longitude" in ds_pr.data_vars:
        lon = ds_pr["longitude"].values
    else:
        lon = ds_pr.coords["longitude"].values
    assert lon.min() > -180.0
    assert lon.max() < 180.0


# ---------------------------------------------------------------------------
# /data/temp — both geodetic and planar geolocation
# ---------------------------------------------------------------------------

def test_temp_data_var(ds_pr):
    assert "temp" in ds_pr.data_vars


def test_temp_both_geolocation_sets(ds_pr):
    """Both geodetic and planar arrays should be present."""
    all_vars = set(ds_pr.data_vars) | set(ds_pr.coords)
    # geodetic
    assert "longitude" in all_vars or "latitude" in all_vars
    # planar
    assert "utm_x" in all_vars or "utm_y" in all_vars


# ---------------------------------------------------------------------------
# DataTree
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dt_geo(geo_store):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return xr.open_datatree(geo_store, engine="xgroup")


def test_datatree_nodes(dt_geo):
    groups = list(dt_geo.groups)
    assert "/data" in groups
    assert "/coords" in groups


def test_datatree_data_has_geolocation(dt_geo):
    data_node = dt_geo["/data"]
    all_vars = set(data_node.data_vars) | set(data_node.coords)
    assert "longitude" in all_vars or "latitude" in all_vars
