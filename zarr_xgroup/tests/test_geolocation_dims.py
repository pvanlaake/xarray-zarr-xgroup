"""
Tests for matching geolocation arrays to the referring array by dimension name
(geolocation convention v1.1.0).
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import zarr

from zarr_xgroup.conventions.geolocation import _xy_dimensions

SPATIAL_UUID = "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"
GEO_UUID = "bb9ee930-8c60-4c47-ad6b-8daa558987ed"
REF_UUID = "d89b30cf-ed8c-43d5-9a16-b492f0cd8786"

SIZES = {"t": 2, "y": 3, "x": 4}


def make_store(path, data_dims=("y", "x"), lon_dims=("y", "x"), lat_dims=("y", "x"),
               lat_shape=None):
    """Store with /g/a (spatial + geolocation) and geolocation arrays /c/lon, /c/lat.
    lon_dims=None writes /c/lon without dimension_names."""
    root = zarr.open_group(str(path), mode="w")
    g, c = root.require_group("g"), root.require_group("c")

    def arr(grp, name, dims, shape=None):
        shape = shape or tuple(SIZES[d] for d in dims)
        return grp.create_array(name, data=np.zeros(shape, "f4"), dimension_names=list(dims))

    a = arr(g, "a", data_dims)
    a.attrs.update({
        "zarr_conventions": [
            {"uuid": SPATIAL_UUID, "name": "spatial"},
            {"uuid": GEO_UUID, "name": "geolocation"},
            {"uuid": REF_UUID, "name": "ref"},
        ],
        "spatial:dimensions": ["y", "x"],
        "spatial:transform": [1.0, 0.0, 0.0, 0.0, -1.0, 3.0],
        "geolocation": {"geodetic": {"x": {"ref": {"node": "../../c/lon"}},
                                     "y": {"ref": {"node": "../../c/lat"}}}},
    })
    if lon_dims is None:
        c.create_array("lon", data=np.zeros((3, 4), "f4"))
    else:
        arr(c, "lon", lon_dims)
    arr(c, "lat", lat_dims, lat_shape)
    return path


def open_with_warnings(path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = xr.open_dataset(path, engine="xgroup", group="/g")
    return ds, [str(x.message) for x in w]


# ---------------------------------------------------------------------------
# Accepted
# ---------------------------------------------------------------------------

def test_matching_dimensions_resolve(tmp_path):
    ds, _ = open_with_warnings(make_store(tmp_path / "s.zarr"))
    assert ds["lon"].dims == ("y", "x")


def test_transposed_dimensions_resolve(tmp_path):
    ds, _ = open_with_warnings(make_store(tmp_path / "s.zarr", lon_dims=("x", "y"),
                                         lat_dims=("x", "y")))
    assert ds["lat"].dims == ("x", "y")


def test_additional_dimension_of_referring_array_resolves(tmp_path):
    ds, _ = open_with_warnings(make_store(tmp_path / "s.zarr", data_dims=("t", "y", "x"),
                                         lon_dims=("t", "y", "x"), lat_dims=("t", "y", "x")))
    assert ds["lon"].dims == ("t", "y", "x")


# ---------------------------------------------------------------------------
# Rejected: warning, and no geolocation variables
# ---------------------------------------------------------------------------

def test_missing_dimension_names_rejected(tmp_path):
    ds, msgs = open_with_warnings(make_store(tmp_path / "s.zarr", lon_dims=None))
    assert any("must have dimension_names" in m for m in msgs)
    assert "lon" not in ds.variables


def test_missing_x_dimension_rejected(tmp_path):
    ds, msgs = open_with_warnings(make_store(tmp_path / "s.zarr", data_dims=("t", "y", "x"),
                                             lat_dims=("t", "y")))
    assert any("X and Y dimensions 'x' and 'y'" in m for m in msgs)
    assert "lat" not in ds.variables


def test_foreign_dimension_rejected(tmp_path):
    ds, msgs = open_with_warnings(make_store(tmp_path / "s.zarr", lat_dims=("t", "y", "x")))
    assert any("not in the referring array: t" in m for m in msgs)
    assert "lat" not in ds.variables


def test_length_mismatch_rejected(tmp_path):
    ds, msgs = open_with_warnings(make_store(tmp_path / "s.zarr", lat_shape=(3, 5)))
    assert any("does not match the lengths" in m for m in msgs)
    assert "lat" not in ds.variables


# ---------------------------------------------------------------------------
# X and Y dimensions from the principal convention
# ---------------------------------------------------------------------------

def node(attrs):
    return SimpleNamespace(attrs=attrs)


def test_xy_dimensions_from_cs_axes_across_crs_objects():
    cs = {"crs": [{"axes": {"range": {"abbreviation": "X"}}},
                  {"axes": {"azimuth": {"abbreviation": "Y"}, "t": {"abbreviation": "T"}}}]}
    assert _xy_dimensions(node({"cs": cs}), None) == ["range", "azimuth"]


def test_xy_dimensions_from_spatial_on_array():
    assert _xy_dimensions(node({"spatial:dimensions": ["lat", "lon"]}), None) == ["lon", "lat"]


def test_xy_dimensions_from_spatial_on_parent_group():
    group = node({"spatial:dimensions": ["row", "col"]})
    assert _xy_dimensions(node({}), group) == ["col", "row"]


def test_xy_dimensions_unknown_without_principal():
    assert _xy_dimensions(node({}), None) is None
    assert _xy_dimensions(node({"cs": {"crs": [{"axes": {"x": {"abbreviation": "X"}}}]}}), None) is None
