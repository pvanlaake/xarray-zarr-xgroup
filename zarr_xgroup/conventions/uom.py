"""
uom convention handler for xarray-zarr-xgroup.

Tier: service
Status: stub — not yet implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_xgroup.conventions.base import (
    ConventionHandler,
    convention_is_declared,
)
from zarr_xgroup.i18n import _

if TYPE_CHECKING:
    import zarr
    import xarray as xr


class UomConventionHandler(ConventionHandler):
    """Handler for the 'uom' service convention."""

    tier = "service"
    name = "uom"
    uuid = None  # TODO: set UUID when assigned

    @staticmethod
    def detect(
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> bool:
        return convention_is_declared(array, name="uom")

    def get_variables(
        self,
        array: zarr.Array,
        group: zarr.Group,
        root: zarr.Group,
    ) -> dict[str, xr.Variable]:
        raise NotImplementedError(
            _("The 'uom' convention handler is not yet implemented.")
        )
