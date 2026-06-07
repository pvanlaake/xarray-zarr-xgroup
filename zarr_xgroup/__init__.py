"""
xarray-zarr-xgroup — XArray backend for cross-group Zarr reference resolution.

Opens Zarr stores with full group hierarchy and cross-group reference
resolution via GeoZarr conventions, producing Dataset and DataTree objects
in which all secondary nodes are present and correctly attached.
"""

from zarr_xgroup.errors import _check_dependencies
_check_dependencies()

__version__ = "0.1.0"

__all__ = ["__version__"]