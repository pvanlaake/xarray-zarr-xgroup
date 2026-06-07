"""
``proj`` service convention handler for xarray-zarr-xgroup.

The ``proj`` convention (UUID: f17cb550-5864-4468-aeb7-f3180cfb622f)
describes coordinate reference systems using an authority identifier, PROJJSON or WKT2, expressed
as ``proj:``-prefixed attributes on Zarr arrays and groups.

As a service convention, ``proj`` produces no coordinate variables.
Its attributes (``proj:code``, ``proj:wkt2``, ``proj:projjson``, etc.)
are preserved automatically in ``Variable.attrs`` by the backend and are
available to downstream tools such as ``rioxarray``.

Detection
---------
An array declares the ``proj`` convention by including a CMO with
UUID ``f17cb550-5864-4468-aeb7-f3180cfb622f`` or name ``"proj"`` in its
``zarr_conventions`` attribute, OR by carrying any attribute whose key
begins with ``"proj:"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_xgroup.conventions.base import (
    ConventionHandler,
    convention_is_declared,
    _PROJ_UUID,
)
from zarr_xgroup.i18n import _

if TYPE_CHECKING:
    import zarr
    import xarray as xr


__all__ = ["ProjConventionHandler"]


class ProjConventionHandler(ConventionHandler):
    """
    Handler for the ``proj`` service convention.

    Behaviour
    ---------
    Produces no coordinate variables. ``proj:``-prefixed attributes on
    the array are preserved in ``Variable.attrs`` automatically by the
    backend and are available to downstream CRS-aware tools.
    """

    tier = "service"
    name = "proj"
    uuid = _PROJ_UUID

    @staticmethod
    def detect(
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> bool:
        """
        Return True if the ``proj`` convention is active on this array.

        Matches on UUID or name in ``zarr_conventions``.
        """
        return convention_is_declared(array, uuid=_PROJ_UUID, name="proj")

    def get_variables(
        self,
        array: zarr.Array,
        group: zarr.Group,
        root: zarr.Group,
    ) -> dict[str, xr.Variable]:
        """
        Return an empty dict.

        The ``proj`` convention carries CRS metadata as array attributes,
        not as coordinate arrays. Those attributes pass through to
        ``Variable.attrs`` automatically.
        """
        return {}
