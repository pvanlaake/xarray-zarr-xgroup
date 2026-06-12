"""
``spatial`` principal convention handler for xarray-zarr-xgroup.

The ``spatial`` convention (UUID: 689b58e2-cf7b-45e0-9fff-9cfc0883d6b4)
describes the relationship between array indices and spatial coordinates
using an affine transform. It covers the two horizontal (X/Y) spatial
axes only.

Attribute inheritance
---------------------
``spatial:`` properties may be defined at the parent group level and
inherited by child arrays that do not define their own values. Array-level
attributes always take precedence over group-level attributes.

This is particularly useful for the ``multiscales`` composition case where
``spatial:dimensions`` and ``spatial:bbox`` are shared across all resolution
levels (defined at group level) while ``spatial:transform`` and
``spatial:shape`` vary per array.

Transform coefficient ordering (rasterio/Affine convention)
-----------------------------------------------------------
``spatial:transform = [a, b, c, d, e, f]``

Maps array index ``(col_index, row_index)`` to coordinate ``(x, y)``:

    x = a * col_index + b * row_index + c
    y = d * col_index + e * row_index + f

For a north-up image with no rotation:
    a = x pixel size (positive)
    b = 0
    c = x coordinate of the top-left corner of the top-left pixel
    d = 0
    e = y pixel size (negative for north-up)
    f = y coordinate of the top-left corner of the top-left pixel

Registration
------------
``"pixel"`` (default) — origin is the top-left **corner** of the top-left
pixel. The centre of pixel ``(row, col)`` is at index ``(col+0.5, row+0.5)``.

``"node"`` — origin is the **centre** of the top-left pixel. The centre
of pixel ``(row, col)`` is at index ``(col, row)`` (no offset).

References
----------
https://github.com/zarr-conventions/spatial/blob/v1/README.md
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_xgroup.conventions.base import (
    ConventionHandler,
    _SPATIAL_UUID,
)
from zarr_xgroup.errors import XGroupReferenceError
from zarr_xgroup.i18n import _

if TYPE_CHECKING:
    import zarr
    import xarray as xr


__all__ = ["SpatialConventionHandler"]

# Name as defined in the v1 schema ("spatial", no trailing colon).
# "spatial:" appears in older examples and geozarr-toolkit output;
# accepted as a legacy fallback.
_SPATIAL_NAMES = ("spatial", "spatial:")


# ---------------------------------------------------------------------------
# Attribute inheritance
# ---------------------------------------------------------------------------

def _get_parent_attrs(array, root) -> dict:
    """
    Return the attributes of the parent group of ``array``.
    Returns an empty dict if the parent cannot be determined or accessed.
    """
    try:
        path = array.path.rstrip("/")
        parent_path = "/".join(path.split("/")[:-1])
        parent = root[parent_path] if parent_path else root
        return dict(parent.attrs)
    except Exception:
        return {}


def _inherited(key: str, array_attrs: dict, parent_attrs: dict):
    """
    Return ``array_attrs[key]`` if present and non-None, otherwise
    ``parent_attrs[key]``. Returns None if absent from both.
    """
    val = array_attrs.get(key)
    if val is not None:
        return val
    return parent_attrs.get(key)


# ---------------------------------------------------------------------------
# Coordinate computation
# ---------------------------------------------------------------------------

def _compute_spatial_coords(
    array,
    transform: list[float],
    dimensions: list[str],
    registration: str,
) -> dict[str, np.ndarray]:
    """
    Compute 1-D coordinate arrays for the X and Y spatial axes.

    Parameters
    ----------
    array : zarr.Array
    transform : list[float]
        [a, b, c, d, e, f] in rasterio/Affine ordering.
    dimensions : list[str]
        [y_dim_name, x_dim_name].
    registration : str
        "pixel" or "node".

    Returns
    -------
    dict mapping dim_name → 1D numpy array of centre coordinates.
    """
    a, b, c, d, e, f = transform
    y_dim, x_dim = dimensions[0], dimensions[1]

    try:
        dim_names = list(array.metadata.dimension_names or [])
    except AttributeError:
        dim_names = list(array.attrs.get("_ARRAY_DIMENSIONS", []))

    def _axis_length(dim_name: str) -> int:
        if dim_name in dim_names:
            return array.shape[dim_names.index(dim_name)]
        raise XGroupReferenceError(
            source_path=getattr(array, "path", "<unknown>"),
            attribute="spatial:dimensions",
            target_path=dim_name,
            reason=_(
                "Dimension '{dim}' listed in spatial:dimensions is not "
                "in the array's dimension_names."
            ).format(dim=dim_name),
        )

    n_rows = _axis_length(y_dim)
    n_cols = _axis_length(x_dim)

    # Index offset: 0.5 for "pixel" (corner-referenced), 0.0 for "node"
    # (centre-referenced).
    offset = 0.5 if registration == "pixel" else 0.0

    col_idx = np.arange(n_cols, dtype=float) + offset
    row_idx = np.arange(n_rows, dtype=float) + offset

    # For a pure affine (non-rotated) grid b==0 and d==0:
    #   x depends only on col_idx → 1-D x coordinate array
    #   y depends only on row_idx → 1-D y coordinate array
    # For a rotated grid the cross-terms require 2-D geolocation arrays.
    # We compute 1-D arrays using only the diagonal terms and warn if
    # rotation is present.

    x_coords = a * col_idx + c
    y_coords = e * row_idx + f

    return {x_dim: x_coords.astype("f8"), y_dim: y_coords.astype("f8")}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class SpatialConventionHandler(ConventionHandler):
    """
    Handler for the ``spatial`` principal convention.

    Reads ``spatial:transform``, ``spatial:dimensions``, and
    ``spatial:registration`` from the array attributes — falling back to
    the parent group's attributes for any property not defined on the
    array itself — and computes 1-D coordinate arrays for the X and Y
    spatial axes.
    """

    tier = "principal"
    name = "spatial"
    uuid = _SPATIAL_UUID

    @staticmethod
    def detect(
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> bool:
        """
        Return True if the ``spatial`` convention is declared on this array
        or its parent group.

        Matches on UUID or name in ``zarr_conventions`` on either the array
        or the parent group. ``"spatial:"`` (trailing colon) is accepted as
        a legacy fallback.
        """
        def _matches(attrs: dict) -> bool:
            for cmo in attrs.get("zarr_conventions", []):
                if not isinstance(cmo, dict):
                    continue
                if cmo.get("uuid") == _SPATIAL_UUID:
                    return True
                if cmo.get("name") in _SPATIAL_NAMES:
                    return True
            return False

        if _matches(dict(array.attrs)):
            return True

        # Check parent group
        try:
            path = array.path.rstrip("/")
            parent_path = "/".join(path.split("/")[:-1])
            parent = root[parent_path] if parent_path else root
            if _matches(dict(parent.attrs)):
                return True
        except Exception:
            pass

        return False

    def get_variables(
        self,
        array: zarr.Array,
        group: zarr.Group,
        root: zarr.Group,
    ) -> dict[str, xr.Variable]:
        """
        Compute and return coordinate variables for the X and Y spatial axes.

        Array-level ``spatial:`` attributes take precedence over group-level
        attributes. Missing array-level attributes are inherited from the
        parent group.
        """
        import xarray as xr

        source_path = getattr(array, "path", "<unknown>")
        attrs = dict(array.attrs)
        parent_attrs = _get_parent_attrs(array, root)

        transform    = _inherited("spatial:transform",      attrs, parent_attrs)
        dimensions   = _inherited("spatial:dimensions",     attrs, parent_attrs)
        registration = _inherited("spatial:registration",   attrs, parent_attrs) or "pixel"
        transform_type = _inherited("spatial:transform_type", attrs, parent_attrs) or "affine"
        bbox         = _inherited("spatial:bbox",           attrs, parent_attrs)

        if transform is None or dimensions is None:
            return {}

        if transform_type != "affine":
            warnings.warn(
                _(
                    "Array '{path}': spatial:transform_type '{tt}' is not "
                    "supported. Only 'affine' is currently implemented. "
                    "Skipping coordinate computation."
                ).format(path=source_path, tt=transform_type),
                RuntimeWarning,
                stacklevel=2,
            )
            return {}

        if len(transform) != 6:
            warnings.warn(
                _(
                    "Array '{path}': spatial:transform must have exactly 6 "
                    "elements, got {n}. Skipping coordinate computation."
                ).format(path=source_path, n=len(transform)),
                RuntimeWarning,
                stacklevel=2,
            )
            return {}

        if len(dimensions) != 2:
            warnings.warn(
                _(
                    "Array '{path}': spatial:dimensions must have exactly 2 "
                    "elements [Y, X], got {n}. Skipping coordinate computation."
                ).format(path=source_path, n=len(dimensions)),
                RuntimeWarning,
                stacklevel=2,
            )
            return {}

        if registration not in ("pixel", "node"):
            warnings.warn(
                _(
                    "Array '{path}': unknown spatial:registration value "
                    "'{reg}', defaulting to 'pixel'."
                ).format(path=source_path, reg=registration),
                RuntimeWarning,
                stacklevel=2,
            )
            registration = "pixel"

        # Warn if rotation terms are non-zero
        b, d = transform[1], transform[3]
        if abs(b) > 1e-10 or abs(d) > 1e-10:
            warnings.warn(
                _(
                    "Array '{path}': spatial:transform has non-zero rotation "
                    "terms (b={b}, d={d}). 1-D coordinate arrays will omit "
                    "the rotation cross-terms and will be approximate. "
                    "Use the geolocation convention for rotated grids."
                ).format(path=source_path, b=b, d=d),
                RuntimeWarning,
                stacklevel=2,
            )

        try:
            coord_data = _compute_spatial_coords(
                array, transform, dimensions, registration
            )
        except XGroupReferenceError:
            raise
        except Exception as exc:
            warnings.warn(
                _(
                    "Array '{path}': failed to compute spatial coordinates: {exc}"
                ).format(path=source_path, exc=exc),
                RuntimeWarning,
                stacklevel=2,
            )
            return {}

        variables: dict[str, xr.Variable] = {}
        for dim_name, data in coord_data.items():
            var_attrs: dict[str, Any] = {}
            if bbox is not None:
                var_attrs["spatial:bbox"] = bbox
            variables[dim_name] = xr.Variable([dim_name], data, var_attrs)

        return variables