"""
Exception and warning hierarchy for xarray-zarr-xgroup.

All exceptions are subclasses of XGroupError. All warnings are
subclasses of XGroupWarning. This allows callers to catch or filter
the entire package's diagnostics with a single base class.

Promoting warnings to errors
-----------------------------
Any warning can be promoted to an error via standard Python warnings
machinery::

    import warnings
    from zarr_xgroup.errors import XGroupNoPrincipalWarning

    warnings.filterwarnings("error", category=XGroupNoPrincipalWarning)

Suppressing warnings
--------------------
::

    warnings.filterwarnings("ignore", category=XGroupWarning)
"""

from __future__ import annotations

from zarr_xgroup.i18n import _


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class XGroupError(Exception):
    """Base class for all xarray-zarr-xgroup exceptions."""


class XGroupWarning(UserWarning):
    """Base class for all xarray-zarr-xgroup warnings."""


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class XGroupNoPrincipalWarning(XGroupWarning):
    """
    Emitted when an array carries no declared principal convention.

    The array is loaded as a plain data variable without secondary node
    resolution. No coordinates are attached.

    To promote this warning to an error::

        import warnings
        from zarr_xgroup.errors import XGroupNoPrincipalWarning
        warnings.filterwarnings("error", category=XGroupNoPrincipalWarning)
    """


# ---------------------------------------------------------------------------
# Reference resolution errors
# ---------------------------------------------------------------------------

class XGroupReferenceError(XGroupError):
    """
    Raised when a declared reference cannot be resolved.

    Attributes
    ----------
    source_path : str
        Absolute path of the array that declared the reference.
    attribute : str
        Name of the attribute containing the unresolved reference.
    target_path : str
        The reference path as declared in the attribute, before resolution.
    reason : str
        Human-readable explanation of why resolution failed.
    """

    def __init__(
        self,
        source_path: str,
        attribute: str,
        target_path: str,
        reason: str,
    ) -> None:
        self.source_path = source_path
        self.attribute = attribute
        self.target_path = target_path
        self.reason = reason
        super().__init__(
            _(
                "Cannot resolve reference in '{source_path}' "
                "(attribute '{attribute}'): '{target_path}' — {reason}"
            ).format(
                source_path=source_path,
                attribute=attribute,
                target_path=target_path,
                reason=reason,
            )
        )


class XGroupStoreError(XGroupError):
    """
    Raised when a cross-store reference target cannot be reached.

    Attributes
    ----------
    source_path : str
        Absolute path of the array that declared the cross-store reference.
    store_uri : str
        URI of the target store that could not be opened.
    reason : str
        Human-readable explanation of why the store could not be reached.
    """

    def __init__(
        self,
        source_path: str,
        store_uri: str,
        reason: str,
    ) -> None:
        self.source_path = source_path
        self.store_uri = store_uri
        self.reason = reason
        super().__init__(
            _(
                "Cannot open cross-store reference from '{source_path}': "
                "store '{store_uri}' — {reason}"
            ).format(
                source_path=source_path,
                store_uri=store_uri,
                reason=reason,
            )
        )


class XGroupPathError(XGroupError):
    """
    Raised when a reference path is syntactically malformed and cannot
    be parsed for resolution.

    Attributes
    ----------
    source_path : str
        Absolute path of the array that declared the malformed reference.
    attribute : str
        Name of the attribute containing the malformed path.
    target_path : str
        The malformed path string as declared in the attribute.
    reason : str
        Human-readable explanation of why the path is malformed.
    """

    def __init__(
        self,
        source_path: str,
        attribute: str,
        target_path: str,
        reason: str,
    ) -> None:
        self.source_path = source_path
        self.attribute = attribute
        self.target_path = target_path
        self.reason = reason
        super().__init__(
            _(
                "Malformed reference path in '{source_path}' "
                "(attribute '{attribute}'): '{target_path}' — {reason}"
            ).format(
                source_path=source_path,
                attribute=attribute,
                target_path=target_path,
                reason=reason,
            )
        )


# ---------------------------------------------------------------------------
# Import-time compatibility check
# ---------------------------------------------------------------------------

def _check_dependencies() -> None:
    """
    Verify that required dependencies meet minimum version requirements.
    Called at package import time from zarr_xgroup/__init__.py.
    Raises ImportError with explicit guidance if any requirement is unmet.
    """
    import importlib.metadata

    requirements = {
        "xarray": "2024.10.0",
        "zarr":   "3.0.0",
        "numpy":  "1.24.0",
    }

    from packaging.version import Version

    for package, min_version in requirements.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise ImportError(
                _(
                    "xarray-zarr-xgroup requires '{package}' but it is not installed. "
                    "Install it with: pip install {package}>={min_version}"
                ).format(package=package, min_version=min_version)
            )
        if Version(installed) < Version(min_version):
            raise ImportError(
                _(
                    "xarray-zarr-xgroup requires {package}>={min_version} "
                    "but {installed} is installed. "
                    "Upgrade with: pip install --upgrade {package}"
                ).format(
                    package=package,
                    min_version=min_version,
                    installed=installed,
                )
            )
