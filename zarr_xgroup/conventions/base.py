"""
Base classes for xarray-zarr-xgroup convention handlers.

Convention handlers interpret the metadata of a Zarr array or group and
return the coordinate variables that should be attached to the XArray
Dataset being constructed.

Two tiers of handlers are defined:

- **Principal handlers** interpret the coordinate structure of an array.
  Exactly one principal convention must be declared per array. A principal
  handler is responsible for returning all coordinate variables implied by
  the array's declared convention.

- **Service handlers** provide auxiliary capabilities (CRS definitions,
  cross-store references, units of measure, geolocation arrays) that
  compose freely with any principal convention. Zero or more service
  handlers may be active for a given array.

All handlers implement the `ConventionHandler` abstract base class.

Adding a new convention
-----------------------
Subclass `ConventionHandler`, implement `detect()` and `get_variables()`,
set `tier` and `name`, then register via the entry point::

    [project.entry-points."zarr_xgroup.conventions"]
    my_convention = "my_package.convention:MyConventionHandler"
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import zarr
    import xarray as xr

from zarr_xgroup.i18n import _


# ---------------------------------------------------------------------------
# Convention Metadata Object helpers
# ---------------------------------------------------------------------------

# Known identifiers for built-in conventions.
# Detection uses uuid as primary key, name as fallback.
_CS_UUID          = "e4dbf0b7-7a00-4ce6-b23e-484292014ab4"
_REF_UUID         = "d89b30cf-ed8c-43d5-9a16-b492f0cd8786"
_SPATIAL_UUID     = "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"
_PROJ_UUID        = "f17cb550-5864-4468-aeb7-f3180cfb622f"
_GEOLOCATION_UUID = "bb9ee930-8c60-4c47-ad6b-8daa558987ed"
_UOM_UUID         = "3bbe438d-df37-49fe-8e2b-739296d46dfb"


def get_declared_conventions(node) -> list[dict]:
    """
    Return the list of Convention Metadata Objects (CMOs) declared in the
    ``zarr_conventions`` attribute of a zarr array or group.

    Returns an empty list if the attribute is absent or malformed.

    Parameters
    ----------
    node : zarr.Array or zarr.Group
        The zarr node to inspect.

    Returns
    -------
    list[dict]
        List of CMO dicts, each containing at least one of ``uuid``,
        ``schema_url``, or ``spec_url``, and optionally ``name``,
        ``version``, and ``description``.
    """
    try:
        conventions = node.attrs.get("zarr_conventions", [])
        if not isinstance(conventions, list):
            return []
        return [c for c in conventions if isinstance(c, dict)]
    except Exception:
        return []


def convention_is_declared(node, *, uuid: str | None = None, name: str | None = None) -> bool:
    """
    Return True if a convention is declared on the node, matching by
    ``uuid`` (preferred) or ``name`` (fallback).

    At least one of ``uuid`` or ``name`` must be supplied.

    Parameters
    ----------
    node : zarr.Array or zarr.Group
    uuid : str, optional
        UUID of the convention to match.
    name : str, optional
        Name of the convention to match (case-sensitive).
    """
    if uuid is None and name is None:
        raise ValueError(_("At least one of 'uuid' or 'name' must be supplied."))

    for cmo in get_declared_conventions(node):
        if uuid is not None and cmo.get("uuid") == uuid:
            return True
        if name is not None and cmo.get("name") == name:
            return True
    return False


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ConventionHandler(ABC):
    """
    Abstract base class for all xarray-zarr-xgroup convention handlers.

    Subclasses must set the class attributes ``tier`` and ``name``, and
    implement the ``detect()`` and ``get_variables()`` methods.

    Class attributes
    ----------------
    tier : {"principal", "service"}
        Whether this is a principal or service convention.
    name : str
        The canonical name of the convention, matching the ``name`` field
        in the Convention Metadata Object.
    uuid : str or None
        The UUID of the convention, if assigned. Used as the primary
        detection key; ``name`` is the fallback.
    """

    tier: Literal["principal", "service"]
    name: str
    uuid: str | None = None

    @staticmethod
    @abstractmethod
    def detect(
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> bool:
        """
        Return True if this convention applies to the given array.

        Detection should be based on the presence of the convention's
        UUID or name in the array's ``zarr_conventions`` attribute.
        The ``root`` and ``group`` arguments provide context for
        conventions that may also be declared at the group level.

        Parameters
        ----------
        root : zarr.Group
            The root group of the Zarr store.
        group : zarr.Group
            The group containing the array.
        array : zarr.Array
            The array being inspected.
        """

    @abstractmethod
    def get_variables(
        self,
        array: zarr.Array,
        group: zarr.Group,
        root: zarr.Group,
    ) -> dict[str, xr.Variable]:
        """
        Interpret this array's convention attributes and return all
        coordinate variables that should be attached to the Dataset.

        The returned dict maps variable names to ``xr.Variable`` objects.
        These will be merged into the variable dict passed to
        ``StoreBackendEntrypoint``, where XArray's coordinate attachment
        logic will find them by name.

        Coordinate arrays that exist as separate Zarr arrays in the store
        should be wrapped lazily (via ``ZarrArrayWrapper``) rather than
        loaded eagerly. Inline coordinate values (``regular``, ``explicit``)
        may be materialized as numpy arrays since they are small by
        definition.

        Parameters
        ----------
        array : zarr.Array
            The primary array being opened.
        group : zarr.Group
            The group containing the array.
        root : zarr.Group
            The root group of the Zarr store, providing access to the
            full store hierarchy for cross-group reference resolution.

        Returns
        -------
        dict[str, xr.Variable]
            Coordinate variables keyed by the name they should appear
            under in the resolved variable dict. An empty dict is valid
            for arrays that declare the convention but have no resolvable
            coordinates (e.g. a service convention with no applicable
            attributes on this particular array).
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ConventionRegistry:
    """
    Registry of convention handlers.

    Built-in handlers are registered at package import time via
    ``register()``. Third-party handlers are discovered from the
    ``zarr_xgroup.conventions`` entry point group.

    The registry maintains separate ordered lists for principal and
    service handlers. Detection order within each tier follows
    registration order.
    """

    def __init__(self) -> None:
        self._principal: list[type[ConventionHandler]] = []
        self._service: list[type[ConventionHandler]] = []

    def register(self, handler_class: type[ConventionHandler]) -> None:
        """
        Register a convention handler class.

        Parameters
        ----------
        handler_class : type[ConventionHandler]
            The handler class to register. Must have ``tier`` set to
            either ``"principal"`` or ``"service"``.

        Raises
        ------
        ValueError
            If ``tier`` is not ``"principal"`` or ``"service"``.
        """
        tier = getattr(handler_class, "tier", None)
        if tier == "principal":
            self._principal.append(handler_class)
        elif tier == "service":
            self._service.append(handler_class)
        else:
            raise ValueError(
                _(
                    "Convention handler '{name}' has invalid tier '{tier}'. "
                    "Must be 'principal' or 'service'."
                ).format(
                    name=getattr(handler_class, "name", repr(handler_class)),
                    tier=tier,
                )
            )

    def detect(
        self,
        root: zarr.Group,
        group: zarr.Group,
        array: zarr.Array,
    ) -> tuple[ConventionHandler | None, list[ConventionHandler]]:
        """
        Detect the active principal and service handlers for an array.

        Returns
        -------
        principal : ConventionHandler or None
            The active principal handler instance, or None if no principal
            convention is declared. The caller is responsible for emitting
            ``XGroupNoPrincipalWarning`` when None is returned.
        services : list[ConventionHandler]
            Zero or more active service handler instances.
        """
        principal = None
        for cls in self._principal:
            if cls.detect(root, group, array):
                principal = cls()
                break

        services = [
            cls()
            for cls in self._service
            if cls.detect(root, group, array)
        ]

        return principal, services

    def load_entry_points(self) -> None:
        """
        Discover and register third-party convention handlers from the
        ``zarr_xgroup.conventions`` entry point group.

        Called once at package import time after built-in handlers are
        registered. Failures to load individual entry points are warned
        about but do not prevent the package from functioning.
        """

        import importlib.metadata

        already_registered = {
            cls.name
            for cls in self._principal + self._service
        }

        try:
            eps = importlib.metadata.entry_points(group="zarr_xgroup.conventions")
        except Exception:
            return

        for ep in eps:
            if ep.name in already_registered:
                continue
            try:
                handler_class = ep.load()
                self.register(handler_class)
            except Exception as exc:
                warnings.warn(
                    _(
                        "Failed to load convention handler from entry point "
                        "'{name}': {exc}"
                    ).format(name=ep.name, exc=exc),
                    RuntimeWarning,
                    stacklevel=2,
                )

# ---------------------------------------------------------------------------
# Package-level registry instance
# ---------------------------------------------------------------------------

#: The global convention registry. Built-in handlers are registered in
#: ``zarr_xgroup/conventions/__init__.py`` after this module is imported.
registry = ConventionRegistry()