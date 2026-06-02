"""
XArray backend entry point for xarray-zarr-xgroup.

This module implements the XArray ``BackendEntrypoint`` that opens Zarr
stores with full cross-group reference resolution. It is registered under
the engine name ``xgroup`` via the ``xarray.backends`` entry point.

Architecture
------------
``XGroupBackendEntrypoint.open_dataset()`` does the following:

1. Opens the full Zarr store at root level, regardless of which group
   the user wants to read. This gives the convention handlers access to
   the complete hierarchy for cross-group reference resolution.

2. For each array in the target group, detects the active principal and
   service convention handlers and calls their ``get_variables()``
   methods to obtain coordinate variables.

3. Assembles a ``ResolvedZarrStore`` — a thin ``AbstractDataStore``
   implementation — whose ``get_variables()`` returns the primary array
   variables plus all resolved coordinate variables.

4. Hands the ``ResolvedZarrStore`` to XArray's ``StoreBackendEntrypoint``
   for standard CF decoding, chunking, and Dataset construction.

The two choke points in XArray's default Zarr backend are bypassed:

- ``ZarrStore._fetch_members()`` only sees arrays in the opened group.
  We open the root store directly and traverse the hierarchy ourselves.

- ``conventions.py`` silently drops the entire ``coordinates`` attribute
  if any referenced name is absent from the variable dict. By the time
  XArray sees the variable dict, all referenced names are present.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
import xarray as xr
from xarray.backends.common import AbstractDataStore, BackendEntrypoint
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.utils import FrozenDict

from zarr_xgroup.conventions.base import registry
from zarr_xgroup.errors import XGroupNoPrincipalWarning
from zarr_xgroup.i18n import _

if TYPE_CHECKING:
    from xarray.core.dataset import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dimension_names(zarr_array: zarr.Array) -> tuple[str, ...]:
    """
    Return the dimension names of a zarr array, supporting both v3
    (``dimension_names`` metadata) and v2 (``_ARRAY_DIMENSIONS`` attr).
    """
    try:
        dims = zarr_array.metadata.dimension_names
        if dims is not None:
            return tuple(dims)
    except AttributeError:
        pass
    dims = zarr_array.attrs.get("_ARRAY_DIMENSIONS")
    if dims is not None:
        return tuple(dims)
    return tuple(f"dim_{i}" for i in range(zarr_array.ndim))


def _wrap_array(zarr_array: zarr.Array, dims: tuple[str, ...]) -> xr.Variable:
    """
    Wrap a zarr array as a lazy ``xr.Variable``.
    """
    from xarray.backends.zarr import ZarrArrayWrapper
    from xarray.core import indexing

    data = indexing.LazilyIndexedArray(ZarrArrayWrapper(zarr_array))
    attrs = {
        k: v for k, v in zarr_array.attrs.items()
        if not k.startswith("_")
    }
    return xr.Variable(dims, data, attrs)


def _rewrite_coordinates_attr(
    variable: xr.Variable,
    resolved_names: set[str],
) -> xr.Variable:
    """
    Rewrite the ``coordinates`` attribute of a variable so that any
    path-like references are replaced by the flat names under which the
    resolved arrays appear in the variable dict.

    This is the mechanism that bypasses XArray's ``conventions.py``
    ``all()`` guard: after rewriting, every name in the ``coordinates``
    string is present in the variable dict.

    For ``cs``-convention arrays, coordinate attachment is handled
    directly by the convention handler and no ``coordinates`` attribute
    rewriting is needed. This rewrite is a safety net for arrays that
    carry a plain ``coordinates`` attribute alongside their convention.
    """
    coords_str = variable.attrs.get("coordinates")
    if not coords_str:
        return variable

    names = coords_str.split()
    rewritten = []
    for name in names:
        # If the name is a path (contains /), use only the basename
        # as the flat name in the variable dict
        flat = name.rsplit("/", 1)[-1] if "/" in name else name
        if flat in resolved_names:
            rewritten.append(flat)
        elif name in resolved_names:
            rewritten.append(name)
        # names not found are silently dropped — they were not resolvable

    if rewritten:
        new_attrs = dict(variable.attrs)
        new_attrs["coordinates"] = " ".join(rewritten)
        return xr.Variable(variable.dims, variable.data, new_attrs, variable.encoding)
    else:
        new_attrs = dict(variable.attrs)
        new_attrs.pop("coordinates", None)
        return xr.Variable(variable.dims, variable.data, new_attrs, variable.encoding)


# ---------------------------------------------------------------------------
# ResolvedZarrStore
# ---------------------------------------------------------------------------

class ResolvedZarrStore(AbstractDataStore):
    """
    A thin ``AbstractDataStore`` that presents a fully resolved variable
    dict to XArray's ``StoreBackendEntrypoint``.

    All cross-group coordinate references have already been resolved by
    the convention handlers before this store is constructed. XArray's
    own CF decoding machinery sees a complete, flat variable dict and
    can attach coordinates without any cross-group awareness.

    Parameters
    ----------
    variables : dict[str, xr.Variable]
        All variables — primary data arrays plus resolved coordinates —
        keyed by their flat name in the Dataset.
    attributes : dict
        Global attributes from the target group's ``.zattrs``.
    dimensions : dict[str, int]
        Dimension name → size mapping derived from all arrays.
    """

    def __init__(
        self,
        variables: dict[str, xr.Variable],
        attributes: dict,
        dimensions: dict[str, int],
    ) -> None:
        self._variables = variables
        self._attributes = attributes
        self._dimensions = dimensions

    def get_variables(self) -> FrozenDict:
        return FrozenDict(self._variables)

    def get_attrs(self) -> dict:
        return self._attributes

    def get_dimensions(self) -> dict[str, int]:
        return self._dimensions

    def get_encoding(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Backend entry point
# ---------------------------------------------------------------------------

class XGroupBackendEntrypoint(BackendEntrypoint):
    """
    XArray backend for cross-group Zarr reference resolution.

    Opens Zarr stores with full group hierarchy awareness and resolves
    all cross-group coordinate references declared by the active
    conventions of each array.

    Usage::

        import xarray as xr

        ds = xr.open_dataset(
            "my_store.zarr",
            engine="xgroup",
            group="/ocean",
        )

        dt = xr.open_datatree(
            "my_store.zarr",
            engine="xgroup",
        )
    """

    description = (
        "Open Zarr stores with full cross-group reference resolution "
        "via GeoZarr conventions"
    )
    url = "https://github.com/R-CF/xarray-zarr-xgroup"
    supports_groups = True
    available = True

    def guess_can_open(self, filename_or_obj) -> bool:
        import os
        if isinstance(filename_or_obj, (str, os.PathLike)):
            return str(filename_or_obj).rstrip("/").endswith(".zarr")
        return False

    # ------------------------------------------------------------------
    # Core resolution logic
    # ------------------------------------------------------------------

    @staticmethod
    def _open_root(
        store_path,
        storage_options: dict | None = None,
    ) -> zarr.Group:
        """Open the Zarr store at root level."""
        kwargs = {}
        if storage_options:
            kwargs["storage_options"] = storage_options
        try:
            return zarr.open(store_path, mode="r", **kwargs)
        except Exception as exc:
            from zarr_xgroup.errors import XGroupStoreError
            raise XGroupStoreError(
                source_path="<root>",
                store_uri=str(store_path),
                reason=str(exc),
            )

    @staticmethod
    def _get_target_group(root: zarr.Group, group: str | None) -> zarr.Group:
        """Navigate to the target group within the store."""
        if group is None or group in ("", "/"):
            return root
        key = group.lstrip("/")
        try:
            return root[key]
        except KeyError:
            from zarr_xgroup.errors import XGroupReferenceError
            raise XGroupReferenceError(
                source_path="<root>",
                attribute="group",
                target_path=group,
                reason=_("Group not found in store."),
            )

    @staticmethod
    def _resolve_group(
        target_group: zarr.Group,
        root: zarr.Group,
        cross_store_storage_options: dict | None,
    ) -> dict[str, xr.Variable]:
        """
        Resolve all arrays in the target group.

        For each array:
        1. Wrap it as a lazy ``xr.Variable`` (the primary data variable).
        2. Detect active convention handlers.
        3. Call each handler's ``get_variables()`` to obtain coordinate
           variables.
        4. Merge everything into a single variable dict.
        """
        variables: dict[str, xr.Variable] = {}

        for arr_name in target_group.array_keys():
            zarr_arr = target_group[arr_name]
            dims = _get_dimension_names(zarr_arr)

            # Primary data variable — always included
            primary = _wrap_array(zarr_arr, dims)
            variables[arr_name] = primary

            # Convention handlers
            principal, services = registry.detect(root, target_group, zarr_arr)

            if principal is None:
                warnings.warn(
                    _(
                        "Array '{path}' has no declared principal convention. "
                        "It will be loaded without coordinate resolution."
                    ).format(path=zarr_arr.path),
                    XGroupNoPrincipalWarning,
                    stacklevel=3,
                )
            else:
                try:
                    coord_vars = principal.get_variables(zarr_arr, target_group, root)
                    variables.update(coord_vars)
                except NotImplementedError:
                    warnings.warn(
                        _(
                            "Principal convention '{name}' is not yet implemented. "
                            "Array '{path}' will be loaded without coordinate resolution."
                        ).format(name=principal.name, path=zarr_arr.path),
                        XGroupNoPrincipalWarning,
                        stacklevel=3,
                    )
                except Exception as exc:
                    warnings.warn(
                        _(
                            "Convention handler '{name}' failed for array '{path}': {exc}"
                        ).format(name=principal.name, path=zarr_arr.path, exc=exc),
                        XGroupNoPrincipalWarning,
                        stacklevel=3,
                    )

            for svc in services:
                try:
                    svc_vars = svc.get_variables(zarr_arr, target_group, root)
                    variables.update(svc_vars)
                except NotImplementedError:
                    pass
                except Exception as exc:
                    warnings.warn(
                        _(
                            "Service convention '{name}' failed for array '{path}': {exc}"
                        ).format(name=svc.name, path=zarr_arr.path, exc=exc),
                        RuntimeWarning,
                        stacklevel=3,
                    )

        # Rewrite path-style coordinates attributes to flat names
        resolved_names = set(variables.keys())
        variables = {
            name: _rewrite_coordinates_attr(var, resolved_names)
            for name, var in variables.items()
        }

        return variables

    @staticmethod
    def _derive_dimensions(variables: dict[str, xr.Variable]) -> dict[str, int]:
        """Derive the dimension name → size mapping from all variables."""
        dims: dict[str, int] = {}
        for var in variables.values():
            for dim, size in zip(var.dims, var.shape):
                if dim in dims and dims[dim] != size:
                    # conflicting sizes — keep the larger (arrays may
                    # include scalar coordinate variables of size 1)
                    dims[dim] = max(dims[dim], size)
                else:
                    dims[dim] = size
        return dims

    # ------------------------------------------------------------------
    # XArray BackendEntrypoint interface
    # ------------------------------------------------------------------

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale: bool = True,
        decode_times: bool = True,
        concat_characters: bool = True,
        decode_coords: bool = True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
        group: str | None = None,
        storage_options: dict | None = None,
        cross_store_storage_options: dict | None = None,
    ) -> Dataset:
        """
        Open a Zarr store as an ``xr.Dataset`` with cross-group
        reference resolution.

        Parameters
        ----------
        filename_or_obj : str or path-like
            Path to the Zarr store.
        group : str, optional
            Path to the group within the store to open as the Dataset
            root. Defaults to the store root.
        storage_options : dict, optional
            Storage options passed to zarr-python when opening the store.
        cross_store_storage_options : dict, optional
            Mapping of URI prefix to storage options for cross-store
            references.
        """
        root = self._open_root(filename_or_obj, storage_options)
        target_group = self._get_target_group(root, group)

        variables = self._resolve_group(
            target_group, root, cross_store_storage_options
        )

        if drop_variables:
            if isinstance(drop_variables, str):
                drop_variables = [drop_variables]
            for name in drop_variables:
                variables.pop(name, None)

        dimensions = self._derive_dimensions(variables)
        attrs = dict(target_group.attrs)

        store = ResolvedZarrStore(variables, attrs, dimensions)
        store_entrypoint = StoreBackendEntrypoint()

        return store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=None,  # already applied above
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

    def open_datatree(
        self,
        filename_or_obj,
        *,
        mask_and_scale: bool = True,
        decode_times: bool = True,
        concat_characters: bool = True,
        decode_coords: bool = True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
        group: str | None = None,
        storage_options: dict | None = None,
        cross_store_storage_options: dict | None = None,
    ) -> xr.DataTree:
        """
        Open a Zarr store as an ``xr.DataTree`` with cross-group
        reference resolution at every node.

        Each group in the hierarchy becomes a ``DataTree`` node. Cross-
        group references are resolved into the Dataset of the node that
        declares them.

        Parameters
        ----------
        filename_or_obj : str or path-like
            Path to the Zarr store.
        group : str, optional
            Root group for the DataTree. Defaults to the store root.
        storage_options : dict, optional
            Storage options passed to zarr-python.
        cross_store_storage_options : dict, optional
            Mapping of URI prefix to storage options for cross-store
            references.
        """
        root = self._open_root(filename_or_obj, storage_options)
        start_group = self._get_target_group(root, group)

        groups_dict: dict[str, xr.Dataset] = {}

        def _visit(zarr_group: zarr.Group, path: str) -> None:
            variables = self._resolve_group(
                zarr_group, root, cross_store_storage_options
            )

            if drop_variables:
                dvars = (
                    [drop_variables]
                    if isinstance(drop_variables, str)
                    else list(drop_variables)
                )
                for name in dvars:
                    variables.pop(name, None)

            dimensions = self._derive_dimensions(variables)
            attrs = dict(zarr_group.attrs)
            store = ResolvedZarrStore(variables, attrs, dimensions)
            store_entrypoint = StoreBackendEntrypoint()

            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=None,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
            groups_dict[path] = ds

            for child_name in zarr_group.group_keys():
                child_group = zarr_group[child_name]
                child_path = f"{path}/{child_name}" if path != "/" else f"/{child_name}"
                _visit(child_group, child_path)

        start_path = "/" if (group is None or group in ("", "/")) else f"/{group.lstrip('/')}"
        _visit(start_group, start_path)

        return xr.DataTree.from_dict(groups_dict)