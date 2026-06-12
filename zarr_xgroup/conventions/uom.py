"""
``uom`` service convention handler for xarray-zarr-xgroup.

The ``uom`` convention defines structured unit-of-measure objects as an
alternative to plain CF UDUNITS strings. It is used by the ``cs``
convention for axes where the unit is complex or non-standard.

As a service convention, ``uom`` produces no coordinate variables.
``uom``-structured unit attributes are preserved in ``Variable.attrs``
automatically by the backend and are available to downstream tools.
Plain string ``units`` attributes from CF are handled by XArray's own
CF decoding layer and are not affected by this handler.

Detection
---------
An array declares the ``uom`` convention by including a CMO with
name ``"uom"`` in its ``zarr_conventions`` attribute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_xgroup.conventions.base import (
    ConventionHandler,
    convention_is_declared,
    _UOM_UUID,
)
from zarr_xgroup.i18n import _

if TYPE_CHECKING:
    import zarr
    import xarray as xr


__all__ = ["UomConventionHandler"]

_UOM_NAME = "uom"


class UomConventionHandler(ConventionHandler):
    """
    Handler for the ``uom`` service convention.

    Behaviour
    ---------
    Produces no coordinate variables. Unit-of-measure attributes are
    preserved in ``Variable.attrs`` automatically by the backend.
    Downstream tools that understand the ``uom`` convention can read
    them from there.

    Plain CF ``units`` string attributes are handled by XArray's own
    CF decoding machinery and are unaffected by this handler.
    """

    tier = "service"
    name = _UOM_NAME
    uuid = _UOM_UUID

    @staticmethod
    def detect(
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> bool:
        """
        Return True if the ``uom`` convention is declared on this array.
        """
        return convention_is_declared(array, name=_UOM_NAME)

    def get_variables(
        self,
        array: zarr.Array,
        group: zarr.Group,
        root: zarr.Group,
    ) -> dict[str, xr.Variable]:
        """
        Return an empty dict.

        Unit-of-measure information is carried as array attributes,
        not as coordinate arrays, and passes through automatically.
        """
        return {}
