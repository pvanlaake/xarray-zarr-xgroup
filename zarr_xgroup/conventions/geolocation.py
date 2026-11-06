"""
``geolocation`` service convention handler for xarray-zarr-xgroup.

The ``geolocation`` convention (UUID: bb9ee930-8c60-4c47-ad6b-8daa558987ed)
provides a standard mechanism to reference geolocation arrays (latitude/
longitude or planar x/y) from data arrays whose spatial coordinates cannot
be expressed by a simple affine transform.

Typical use cases include satellite swath data (MODIS, VIIRS, Sentinel-3),
curvilinear model grids (CORDEX rotated pole, ROMS tripolar), and any array
where the relationship between array indices and geographic coordinates
requires 2-D lookup tables rather than a formula.

This handler is always used in combination with a principal convention
(``cs`` or ``spatial``) — never standalone.

Structure
---------
The ``geolocation`` attribute has the following structure::

    {
      "geolocation": {
        "geodetic": {
          "x": {"ref": {"node": "longitude"}},
          "y": {"ref": {"node": "latitude"}},
          "id": {"proj:code": "EPSG:4326"}    # optional
        },
        "planar": {                             # optional
          "x": {"ref": {"node": "../grid/easting"}},
          "y": {"ref": {"node": "../grid/northing"}},
          "id": {"proj:code": "EPSG:32628"}
        }
      }
    }

The ``x`` and ``y`` fields each contain a ``{"ref": <ref object>}``
wrapper, consistent with other conventions in this ecosystem. For
``geodetic``, ``x`` is longitude and ``y`` is latitude. For ``planar``,
``x`` is easting and ``y`` is northing. These names can be freely
chosen by the data producer.

Dimension matching
------------------
The geolocation arrays MUST have ``dimension_names``. They are matched to
the referring array by name: the ``X`` and ``Y`` dimensions of the referring
array, as identified by the principal convention, MUST appear in each
geolocation array, every other dimension of a geolocation array MUST also be
a dimension of the referring array, and the lengths of matching dimensions
MUST be equal. The order of the dimensions is free.

Variable naming
---------------
Resolved geolocation arrays are added to the variable dict under the
basename of the resolved array path (e.g. ``longitude``, ``latitude``,
``easting``, ``northing``). If both ``geodetic`` and ``planar`` are
present, the basenames of the referenced arrays must differ to avoid
name collisions.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from zarr_xgroup.conventions.base import (
    ConventionHandler,
    convention_is_declared,
    _GEOLOCATION_UUID,
)
from zarr_xgroup.conventions.ref import resolve_ref
from zarr_xgroup.errors import XGroupReferenceError
from zarr_xgroup.i18n import _

if TYPE_CHECKING:
    import zarr
    import xarray as xr


__all__ = ["GeolocationConventionHandler"]

_GEOLOCATION_NAME = "geolocation"


def _dimension_names(arr) -> list:
    """Dimension names of a Zarr array (v3 metadata or v2 attribute)."""
    try:
        names = arr.metadata.dimension_names
    except AttributeError:
        names = arr.attrs.get("_ARRAY_DIMENSIONS")
    return list(names or [])


def _xy_dimensions(array, group) -> list[str] | None:
    """
    Names of the X and Y dimensions of ``array`` as identified by its
    principal convention: the axes with abbreviation ``X`` and ``Y`` in the
    ``cs`` attribute, else ``spatial:dimensions`` (Y, X order) on the array
    or its parent group. Returns None when they cannot be identified.
    """
    attrs = dict(array.attrs)
    cs = attrs.get("cs")
    if isinstance(cs, dict):
        abbr = {ax.get("abbreviation"): dim
                for crs in cs.get("crs", []) if isinstance(crs, dict)
                for dim, ax in (crs.get("axes") or {}).items() if isinstance(ax, dict)}
        xy = [abbr.get("X"), abbr.get("Y")]
    else:
        dims = attrs.get("spatial:dimensions")
        if dims is None and group is not None:
            dims = dict(group.attrs).get("spatial:dimensions")
        xy = [dims[1], dims[0]] if dims and len(dims) == 2 else [None, None]
    return xy if all(xy) else None


def _resolve_arrays_object(
    arrays_obj: dict,
    kind: str,
    array,
    group,
    root,
    source_path: str,
    variables: dict,
) -> None:
    """
    Resolve the x and y ref objects in an ``arrays`` object and add
    the resulting variables to ``variables`` in place.

    Parameters
    ----------
    arrays_obj : dict
        The ``geodetic`` or ``planar`` object with ``x``, ``y``, and
        optional ``id`` fields. Each of ``x`` and ``y`` is a
        ``{"ref": <ref object>}`` wrapper.
    kind : str
        ``"geodetic"`` or ``"planar"`` — used in error messages only.
    """
    import zarr as zarr_mod
    import xarray as xr
    from xarray.backends.zarr import ZarrArrayWrapper
    from xarray.core import indexing

    ref_dims = _dimension_names(array)
    xy_dims = _xy_dimensions(array, group)

    for axis in ("x", "y"):
        wrapper = arrays_obj.get(axis)
        if wrapper is None:
            raise XGroupReferenceError(
                source_path=source_path,
                attribute=f"geolocation/{kind}/{axis}",
                target_path="<missing>",
                reason=_(
                    "Required '{axis}' field missing from "
                    "geolocation/{kind} object."
                ).format(axis=axis, kind=kind),
            )

        # Unwrap {"ref": <ref object>}
        ref_obj = wrapper.get("ref") if isinstance(wrapper, dict) else None
        if ref_obj is None:
            raise XGroupReferenceError(
                source_path=source_path,
                attribute=f"geolocation/{kind}/{axis}",
                target_path=repr(wrapper),
                reason=_(
                    "geolocation/{kind}/{axis} must be a "
                    "{{\"ref\": {{...}}}} object."
                ).format(kind=kind, axis=axis),
            )

        resolved = resolve_ref(
            ref_obj,
            referencing_group=group,
            referencing_path=source_path,
            store_root=root,
            source_path=source_path,
            attribute=f"geolocation/{kind}/{axis}/ref",
        )

        if not isinstance(resolved, zarr_mod.Array):
            raise XGroupReferenceError(
                source_path=source_path,
                attribute=f"geolocation/{kind}/{axis}/ref",
                target_path=ref_obj.get("node", "<unknown>"),
                reason=_(
                    "geolocation/{kind}/{axis} must reference a Zarr array, "
                    "got {type}."
                ).format(kind=kind, axis=axis, type=type(resolved).__name__),
            )

        # Use the basename of the resolved array path as the variable name
        var_name = resolved.path.rstrip("/").rsplit("/", 1)[-1]

        dims = _dimension_names(resolved)
        reason = None
        if len(dims) != resolved.ndim or any(d is None for d in dims):
            reason = _("geolocation arrays must have dimension_names.")
        elif xy_dims and not all(d in dims for d in xy_dims):
            reason = _(
                "geolocation array must have the X and Y dimensions "
                "'{x}' and '{y}' of the referring array."
            ).format(x=xy_dims[0], y=xy_dims[1])
        elif any(d not in ref_dims for d in dims):
            reason = _(
                "geolocation array has dimensions that are not in the "
                "referring array: {dims}."
            ).format(dims=", ".join(d for d in dims if d not in ref_dims))
        elif any(n != array.shape[ref_dims.index(d)] for d, n in zip(dims, resolved.shape)):
            reason = _(
                "geolocation array does not match the lengths of the "
                "dimensions of the referring array."
            )
        if reason is not None:
            raise XGroupReferenceError(
                source_path=source_path,
                attribute=f"geolocation/{kind}/{axis}/ref",
                target_path=ref_obj.get("node", "<unknown>"),
                reason=reason,
            )

        attrs = dict(resolved.attrs)
        # Record the geolocation role for downstream tools
        role = "longitude" if (kind == "geodetic" and axis == "x") else \
               "latitude"  if (kind == "geodetic" and axis == "y") else \
               "easting"   if (kind == "planar"   and axis == "x") else \
               "northing"
        attrs["geolocation_role"] = f"{kind}_{role}"

        lazy = indexing.LazilyIndexedArray(ZarrArrayWrapper(resolved))
        variables[var_name] = xr.Variable(dims, lazy, attrs)


class GeolocationConventionHandler(ConventionHandler):
    """
    Handler for the ``geolocation`` service convention.

    Resolves ``geodetic`` and/or ``planar`` geolocation array references
    and returns them as lazy coordinate variables keyed by the basename
    of the referenced array path.

    This handler is a service convention — it operates alongside a
    principal convention (``cs`` or ``spatial``) and does not replace it.
    """

    tier = "service"
    name = _GEOLOCATION_NAME
    uuid = _GEOLOCATION_UUID

    @staticmethod
    def detect(
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> bool:
        """
        Return True if the ``geolocation`` convention is declared on this array.
        """
        return convention_is_declared(
            array, uuid=_GEOLOCATION_UUID, name=_GEOLOCATION_NAME
        )

    def get_variables(
        self,
        array: zarr.Array,
        group: zarr.Group,
        root: zarr.Group,
    ) -> dict[str, xr.Variable]:
        """
        Resolve geolocation array references and return them as variables.

        Processes both ``geodetic`` and ``planar`` sub-objects if present.
        Each resolved array is returned as a lazy ``xr.Variable`` keyed by
        the basename of its path in the store.
        """
        source_path = getattr(array, "path", "<unknown>")
        geo_obj = dict(array.attrs).get("geolocation")

        if geo_obj is None:
            return {}

        variables: dict[str, xr.Variable] = {}

        for kind in ("geodetic", "planar"):
            arrays_obj = geo_obj.get(kind)
            if arrays_obj is None:
                continue
            try:
                _resolve_arrays_object(
                    arrays_obj,
                    kind=kind,
                    array=array,
                    group=group,
                    root=root,
                    source_path=source_path,
                    variables=variables,
                )
            except XGroupReferenceError:
                raise
            except Exception as exc:
                warnings.warn(
                    _(
                        "Array '{path}': failed to resolve "
                        "geolocation/{kind} arrays: {exc}"
                    ).format(path=source_path, kind=kind, exc=exc),
                    RuntimeWarning,
                    stacklevel=2,
                )

        return variables