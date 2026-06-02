"""
``ref`` service convention handler for xarray-zarr-xgroup.

The ``ref`` convention (UUID: d89b30cf-ed8c-43d5-9a16-b492f0cd8786)
defines a standard mechanism for referencing external nodes — arrays,
groups, or attribute fragments — from within a Zarr node's attributes.

As a service convention, ``ref`` does not produce coordinate variables
directly. It is invoked by principal convention handlers (such as ``cs``)
when they encounter ``ref`` objects embedded in their convention's
attribute structure.

The primary public API of this module is the ``resolve_ref()`` function,
re-exported here for convenience so that other handlers can import it
from a single location::

    from zarr_xgroup.conventions.ref import resolve_ref

    node_or_value = resolve_ref(
        ref_obj,
        referencing_group=group,
        store_root=root,
        source_path=array.path,
        attribute="cs/crs/0/ref",
    )

See ``zarr_xgroup.resolver`` for full documentation of the resolution
rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_xgroup.conventions.base import (
    ConventionHandler,
    convention_is_declared,
    _REF_UUID,
)
from zarr_xgroup.resolver import resolve_ref  # re-exported for convenience
from zarr_xgroup.i18n import _

if TYPE_CHECKING:
    import zarr
    import xarray as xr


__all__ = ["RefConventionHandler", "resolve_ref"]


class RefConventionHandler(ConventionHandler):
    """
    Handler for the ``ref`` service convention.

    Detection
    ---------
    An array declares the ``ref`` convention by including a CMO with
    UUID ``d89b30cf-ed8c-43d5-9a16-b492f0cd8786`` or name ``"ref"``
    in its ``zarr_conventions`` attribute.

    Behaviour
    ---------
    This handler produces no coordinate variables itself. Its presence
    signals to the backend and to other handlers that ``ref`` objects
    found in the node's attributes should be resolved using the
    ``resolve_ref()`` function from ``zarr_xgroup.resolver``.

    Other handlers (e.g. ``cs``) check whether the ``ref`` convention
    is active before attempting to resolve ``ref`` objects they
    encounter in the primary convention's attribute structure.
    """

    tier = "service"
    name = "ref"
    uuid = _REF_UUID

    @staticmethod
    def detect(
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> bool:
        """
        Return True if the ``ref`` convention is declared on this array.

        Detection checks the ``zarr_conventions`` attribute of the array
        for a CMO matching by UUID (preferred) or name.
        """
        return convention_is_declared(array, uuid=_REF_UUID, name="ref")

    def get_variables(
        self,
        array: zarr.Array,
        group: zarr.Group,
        root: zarr.Group,
    ) -> dict[str, xr.Variable]:
        """
        Return an empty dict.

        The ``ref`` convention produces no coordinate variables directly.
        Reference resolution is performed by calling ``resolve_ref()``
        from within other convention handlers.
        """
        return {}