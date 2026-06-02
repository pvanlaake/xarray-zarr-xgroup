"""
proj convention handler for xarray-zarr-xgroup.

Tier: service
Status: stub — not yet implemented.
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


class ProjConventionHandler(ConventionHandler):
    """Handler for the 'proj' service convention."""

    tier = "service"
    name = "proj"
    uuid = "f17cb550-5864-4468-aeb7-f3180cfb622f"

    @staticmethod
    def detect(
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> bool:
        return convention_is_declared(array, uuid=_PROJ_UUID, name="proj")

    def get_variables(
        self,
        array: zarr.Array,
        group: zarr.Group,
        root: zarr.Group,
    ) -> dict[str, xr.Variable]:
        raise NotImplementedError(
            _("The 'proj' convention handler is not yet implemented.")
        )
