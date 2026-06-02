"""
spatial convention handler for xarray-zarr-xgroup.

Tier: principal
Status: stub — not yet implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_xgroup.conventions.base import (
    ConventionHandler,
    convention_is_declared,
    _SPATIAL_UUID,
)
from zarr_xgroup.i18n import _

if TYPE_CHECKING:
    import zarr
    import xarray as xr


class SpatialConventionHandler(ConventionHandler):
    """Handler for the 'spatial' principal convention."""

    tier = "principal"
    name = "spatial"
    uuid = "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"

    @staticmethod
    def detect(
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> bool:
        return convention_is_declared(array, uuid=_SPATIAL_UUID, name="spatial")

    def get_variables(
        self,
        array: zarr.Array,
        group: zarr.Group,
        root: zarr.Group,
    ) -> dict[str, xr.Variable]:
        raise NotImplementedError(
            _("The 'spatial' convention handler is not yet implemented.")
        )
