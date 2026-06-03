"""
``cs`` principal convention handler for xarray-zarr-xgroup.

The ``cs`` (Coordinate Set) convention (UUID: e4dbf0b7-7a00-4ce6-b23e-484292014ab4)
attaches coordinate values to the dimensions of a Zarr array using a
structured JSON schema. It is based on the OGC standard "Referencing by
Coordinates" and supports regular, explicit, and external coordinate values,
boundary values, parametric vertical coordinates, and curvilinear geolocation
arrays.

This handler traverses the ``cs`` attribute structure of an array, resolves
all ``ref`` objects it encounters, and returns the resulting coordinate
variables as a dict of ``xr.Variable`` objects keyed by name.

Variable naming convention
--------------------------
- Primary coordinate for axis ``dim``: ``dim``
- Boundary array for axis ``dim``: ``dim_bounds``
- Parametric term ``term`` for axis ``dim``: ``dim_pterm_{term}``
- Additional coordinate set ``n`` for axis ``dim``: ``dim_{n}``

The ``geolocation`` arrays are returned under the names declared by the
``geolocation`` convention handler, not by this handler directly.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_xgroup.conventions.base import (
    ConventionHandler,
    convention_is_declared,
    _CS_UUID,
    _REF_UUID,
)
from zarr_xgroup.conventions.ref import resolve_ref
from zarr_xgroup.errors import XGroupNoPrincipalWarning, XGroupReferenceError
from zarr_xgroup.i18n import _

if TYPE_CHECKING:
    import zarr
    import xarray as xr


__all__ = ["CsConventionHandler"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_is_declared(array) -> bool:
    """Return True if the ``ref`` convention is declared on this array."""
    return convention_is_declared(array, uuid=_REF_UUID, name="ref")


def _make_regular(values_obj: dict, length: int) -> np.ndarray:
    """
    Materialise a ``regular`` values specification into a numpy array.

    Parameters
    ----------
    values_obj : dict
        The ``regular`` field value: ``[start, increment]``.
    length : int
        Number of coordinate values to generate.
    """
    start, increment = values_obj[0], values_obj[1]
    return np.array([start + i * increment for i in range(length)], dtype=float)


def _resolve_values(
    values_obj: dict,
    *,
    axis_name: str,
    array,
    group,
    root,
    ref_declared: bool,
    source_path: str,
    dim_length: int,
) -> np.ndarray | Any:
    """
    Resolve a ``values`` object to either a numpy array (inline) or a
    lazy zarr array reference (external).

    Returns the raw data — the caller wraps it in ``xr.Variable``.
    """
    if "regular" in values_obj:
        return _make_regular(values_obj["regular"], dim_length)

    if "explicit" in values_obj:
        return np.array(values_obj["explicit"])

    if "external" in values_obj:
        ext = values_obj["external"]
        ref_obj = ext.get("ref")
        if ref_obj is None:
            raise XGroupReferenceError(
                source_path=source_path,
                attribute=f"cs/axes/{axis_name}/values/external",
                target_path="<missing>",
                reason=_("'external' object is missing 'ref' field."),
            )
        if not ref_declared:
            warnings.warn(
                _(
                    "Array '{path}' uses 'external' coordinate values but "
                    "does not declare the 'ref' convention. "
                    "Reference resolution will be attempted anyway."
                ).format(path=source_path),
                XGroupNoPrincipalWarning,
                stacklevel=4,
            )
        return resolve_ref(
            ref_obj,
            referencing_group=group,
            store_root=root,
            source_path=source_path,
            attribute=f"cs/axes/{axis_name}/values/external/ref",
        )

    raise XGroupReferenceError(
        source_path=source_path,
        attribute=f"cs/axes/{axis_name}/values",
        target_path="<unknown>",
        reason=_("'values' object must have one of: 'regular', 'explicit', 'external'."),
    )


def _resolve_boundaries(
    boundaries_obj: dict,
    *,
    axis_name: str,
    array,
    group,
    root,
    ref_declared: bool,
    source_path: str,
    dim_length: int,
) -> np.ndarray | Any | None:
    """
    Resolve a ``boundaries`` object. Returns None if not present.
    """
    if boundaries_obj is None:
        return None

    if "regular" in boundaries_obj:
        lo, hi = boundaries_obj["regular"]
        return np.array([[lo, hi]] * dim_length, dtype=float)

    if "external" in boundaries_obj:
        ext = boundaries_obj["external"]
        ref_obj = ext.get("ref")
        if ref_obj is None:
            raise XGroupReferenceError(
                source_path=source_path,
                attribute=f"cs/axes/{axis_name}/boundaries/external",
                target_path="<missing>",
                reason=_("'external' object is missing 'ref' field."),
            )
        return resolve_ref(
            ref_obj,
            referencing_group=group,
            store_root=root,
            source_path=source_path,
            attribute=f"cs/axes/{axis_name}/boundaries/external/ref",
        )

    return None


def _array_dim_length(array, dim_name: str) -> int:
    """Return the length of a named dimension in the array's shape."""
    try:
        dim_names = list(array.metadata.dimension_names or [])
    except AttributeError:
        dim_names = list(array.attrs.get("_ARRAY_DIMENSIONS", []))

    if dim_name in dim_names:
        idx = dim_names.index(dim_name)
        return array.shape[idx]
    # scalar axis not in dimension_names — length is 1
    return 1


def _process_coordinates_set(
    coords_obj: dict,
    *,
    axis_name: str,
    coord_idx: int,
    array,
    group,
    root,
    ref_declared: bool,
    source_path: str,
    variables: dict,
) -> None:
    """
    Process one ``coordinates`` object within an axis and add the
    resulting variables to ``variables`` in place.
    """
    import xarray as xr

    dim_length = _array_dim_length(array, axis_name)

    values_obj = coords_obj.get("values")
    if values_obj is None:
        # ordinal axis — no coordinate variable to attach
        return

    data = _resolve_values(
        values_obj,
        axis_name=axis_name,
        array=array,
        group=group,
        root=root,
        ref_declared=ref_declared,
        source_path=source_path,
        dim_length=dim_length,
    )

    # Build attrs for this coordinate variable
    attrs: dict[str, Any] = {}
    if "direction" in coords_obj:
        attrs["direction"] = coords_obj["direction"]
    if "unit" in coords_obj:
        attrs["units"] = coords_obj["unit"]
    if "time" in coords_obj:
        time_obj = coords_obj["time"]
        attrs["units"] = f"{time_obj['unit']} since {time_obj['epoch']}"
        if "calendar" in time_obj:
            attrs["calendar"] = time_obj["calendar"]
    if "attributes" in coords_obj:
        attrs.update(coords_obj["attributes"])

    # Variable name: primary coordinate uses axis name,
    # additional sets get a numeric suffix
    var_name = axis_name if coord_idx == 0 else f"{axis_name}_{coord_idx}"

    # Wrap zarr arrays lazily; numpy arrays are already materialised
    import zarr as zarr_mod
    if isinstance(data, zarr_mod.Array):
        from xarray.backends.zarr import ZarrArrayWrapper
        from xarray.core import indexing
        lazy = indexing.LazilyIndexedArray(ZarrArrayWrapper(data))
        raw_dims = data.metadata.dimension_names
        dims = list(raw_dims) if raw_dims else ([] if data.ndim == 0 else [axis_name])
        variables[var_name] = xr.Variable(dims, lazy, attrs)
    else:
        dims = [axis_name] if data.ndim >= 1 else []
        variables[var_name] = xr.Variable(dims, data, attrs)

    # Boundaries
    bounds_obj = coords_obj.get("boundaries")
    if bounds_obj is not None:
        bounds_data = _resolve_boundaries(
            bounds_obj,
            axis_name=axis_name,
            array=array,
            group=group,
            root=root,
            ref_declared=ref_declared,
            source_path=source_path,
            dim_length=dim_length,
        )
        if bounds_data is not None:
            bounds_name = f"{var_name}_bounds"
            if isinstance(bounds_data, zarr_mod.Array):
                from xarray.backends.zarr import ZarrArrayWrapper
                from xarray.core import indexing
                lazy = indexing.LazilyIndexedArray(ZarrArrayWrapper(bounds_data))
                dims = list(bounds_data.metadata.dimension_names or [axis_name, "bounds"])
                variables[bounds_name] = xr.Variable(dims, lazy, {})
            else:
                variables[bounds_name] = xr.Variable(
                    [axis_name, "bounds"], bounds_data, {}
                )
            # record bounds relationship in coordinate attrs
            variables[var_name].attrs["bounds"] = bounds_name

    # Parametric terms
    parametric_obj = coords_obj.get("parametric")
    if parametric_obj is not None:
        formula = parametric_obj.get("formula", "")
        terms = parametric_obj.get("terms", {})
        for term_name, term_values_obj in terms.items():
            term_data = _resolve_values(
                term_values_obj,
                axis_name=axis_name,
                array=array,
                group=group,
                root=root,
                ref_declared=ref_declared,
                source_path=source_path,
                dim_length=dim_length,
            )
            pterm_name = f"{var_name}_pterm_{term_name}"
            term_attrs = {"parametric_formula": formula, "parametric_term": term_name}
            if isinstance(term_data, zarr_mod.Array):
                from xarray.backends.zarr import ZarrArrayWrapper
                from xarray.core import indexing
                lazy = indexing.LazilyIndexedArray(ZarrArrayWrapper(term_data))
                raw_tdims = term_data.metadata.dimension_names
                tdims = list(raw_tdims) if raw_tdims else ([] if term_data.ndim == 0 else [axis_name])
                variables[pterm_name] = xr.Variable(tdims, lazy, term_attrs)
            else:
                tdims = [axis_name] if term_data.ndim >= 1 else []
                variables[pterm_name] = xr.Variable(tdims, term_data, term_attrs)


def _process_crs(
    crs_obj: dict,
    *,
    array,
    group,
    root,
    ref_declared: bool,
    source_path: str,
    variables: dict,
) -> None:
    """
    Process one ``crs`` object (inline or already resolved from a ref)
    and add coordinate variables to ``variables`` in place.
    """
    axes = crs_obj.get("axes", {})
    if not isinstance(axes, dict):
        return

    for axis_name, axis_obj in axes.items():
        if not isinstance(axis_obj, dict):
            continue

        coordinates_list = axis_obj.get("coordinates")
        if coordinates_list is None:
            # ordinal axis — no coordinates to attach
            continue

        for coord_idx, coords_obj in enumerate(coordinates_list):
            if not isinstance(coords_obj, dict):
                continue
            _process_coordinates_set(
                coords_obj,
                axis_name=axis_name,
                coord_idx=coord_idx,
                array=array,
                group=group,
                root=root,
                ref_declared=ref_declared,
                source_path=source_path,
                variables=variables,
            )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class CsConventionHandler(ConventionHandler):
    """
    Handler for the ``cs`` principal convention.

    Traverses the ``cs`` attribute structure of an array, resolves all
    ``ref`` objects it encounters (for external coordinate arrays and
    cross-group CRS references), and returns coordinate variables for
    all declared axes.
    """

    tier = "principal"
    name = "cs"
    uuid = _CS_UUID

    @staticmethod
    def detect(
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> bool:
        """
        Return True if the ``cs`` convention is declared on this array.
        """
        return convention_is_declared(array, uuid=_CS_UUID, name="cs")

    def get_variables(
        self,
        array: zarr.Array,
        group: zarr.Group,
        root: zarr.Group,
    ) -> dict[str, xr.Variable]:
        """
        Traverse the ``cs`` attribute and return all coordinate variables.

        Steps
        -----
        1. Read ``array.attrs["cs"]`` to get the coordinate set object.
        2. Iterate over ``cs.crs`` — each item is either an inline CRS
           object or a ``ref`` object pointing to a CRS in a group.
        3. For each CRS, iterate over ``axes`` (keyed by dimension name).
        4. For each axis, iterate over ``coordinates`` and resolve
           ``values`` and ``boundaries`` (regular, explicit, or external).
        5. For parametric axes, resolve each formula term.
        6. Return all resulting ``xr.Variable`` objects.
        """
        source_path = getattr(array, "path", "<unknown>")
        ref_declared = _ref_is_declared(array)

        cs_obj = dict(array.attrs).get("cs")
        if cs_obj is None:
            return {}

        crs_list = cs_obj.get("crs")
        if not crs_list:
            return {}

        variables: dict[str, xr.Variable] = {}

        for crs_item in crs_list:
            if not isinstance(crs_item, dict):
                continue

            # Cross-group CRS reference
            if "ref" in crs_item and len(crs_item) == 1:
                if not ref_declared:
                    warnings.warn(
                        _(
                            "Array '{path}' references a CRS via 'ref' but "
                            "does not declare the 'ref' convention."
                        ).format(path=source_path),
                        XGroupNoPrincipalWarning,
                        stacklevel=2,
                    )
                try:
                    resolved = resolve_ref(
                        crs_item["ref"],
                        referencing_group=group,
                        store_root=root,
                        source_path=source_path,
                        attribute="cs/crs/ref",
                    )
                except Exception as exc:
                    warnings.warn(
                        _(
                            "Could not resolve CRS ref in '{path}': {exc}"
                        ).format(path=source_path, exc=exc),
                        XGroupNoPrincipalWarning,
                        stacklevel=2,
                    )
                    continue

                # resolved is either a dict (attribute fragment) or a node
                if isinstance(resolved, dict):
                    crs_obj = resolved
                else:
                    # unlikely but handle gracefully
                    continue

                _process_crs(
                    crs_obj,
                    array=array,
                    group=group,
                    root=root,
                    ref_declared=ref_declared,
                    source_path=source_path,
                    variables=variables,
                )

            else:
                # Inline CRS object
                _process_crs(
                    crs_item,
                    array=array,
                    group=group,
                    root=root,
                    ref_declared=ref_declared,
                    source_path=source_path,
                    variables=variables,
                )

        return variables