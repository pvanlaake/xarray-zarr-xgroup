"""
Convention handler registry initialisation for xarray-zarr-xgroup.

This module imports all built-in convention handlers, registers them with
the package-level registry in the correct order, and discovers any
third-party handlers registered via the ``zarr_xgroup.conventions``
entry point group.

Registration order
------------------
Principal handlers are registered first, in priority order. If two
principal handlers both claim an array (which should not happen in a
well-formed store), the first registered handler wins.

Service handlers are registered after principal handlers. All active
service handlers are applied to every array regardless of which principal
handler is active.

Built-in principal conventions (in registration order):
    1. ``cs``      — Coordinate Set convention
    2. ``spatial`` — Spatial coordinate convention

Built-in service conventions (in registration order):
    1. ``ref``         — Cross-node and cross-store references
    2. ``proj``        — CRS description via PROJJSON or WKT2
    3. ``uom``         — Unit of measure definitions
    4. ``geolocation`` — Geolocation arrays for curvilinear grids
"""

from __future__ import annotations

from zarr_xgroup.conventions.base import registry

# ---------------------------------------------------------------------------
# Principal conventions
# ---------------------------------------------------------------------------

from zarr_xgroup.conventions.cs import CsConventionHandler
from zarr_xgroup.conventions.spatial import SpatialConventionHandler

registry.register(CsConventionHandler)
registry.register(SpatialConventionHandler)

# ---------------------------------------------------------------------------
# Service conventions
# ---------------------------------------------------------------------------

from zarr_xgroup.conventions.ref import RefConventionHandler
from zarr_xgroup.conventions.proj import ProjConventionHandler
from zarr_xgroup.conventions.uom import UomConventionHandler
from zarr_xgroup.conventions.geolocation import GeolocationConventionHandler

registry.register(RefConventionHandler)
registry.register(ProjConventionHandler)
registry.register(UomConventionHandler)
registry.register(GeolocationConventionHandler)

# ---------------------------------------------------------------------------
# Third-party conventions via entry points
# ---------------------------------------------------------------------------

registry.load_entry_points()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "registry",
    "CsConventionHandler",
    "SpatialConventionHandler",
    "RefConventionHandler",
    "ProjConventionHandler",
    "UomConventionHandler",
    "GeolocationConventionHandler",
]