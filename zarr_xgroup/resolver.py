"""
Path resolution for xarray-zarr-xgroup.

This module implements resolution of ``ref`` objects as defined by the
``ref`` convention (UUID: d89b30cf-ed8c-43d5-9a16-b492f0cd8786).

A ``ref`` object has the following fields:

- ``node`` (required): path to a group or array, either relative to the
  referencing node's group or absolute from the store root.
- ``attribute`` (optional): RFC 6901 JSON pointer into the ``zarr.json``
  of the target node.
- ``uri`` (optional): URI of an external Zarr store. If present, ``node``
  is an absolute path within that store.

Path resolution rules
---------------------
1. If ``uri`` is present, open the external store and resolve ``node``
   as an absolute path within it.
2. If ``node`` begins with ``/``, resolve as an absolute path from the
   store root.
3. Otherwise, resolve as a relative path from the referencing node's
   group, interpreting ``..`` as parent group traversal per RFC 3986.
4. If ``attribute`` is present, apply it as a JSON pointer to the
   target node's attributes and return the pointed-to value.
5. If ``attribute`` is absent, return the target node itself
   (``zarr.Array`` or ``zarr.Group``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import zarr

from zarr_xgroup.errors import XGroupPathError, XGroupReferenceError, XGroupStoreError
from zarr_xgroup.i18n import _

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# JSON Pointer (RFC 6901)
# ---------------------------------------------------------------------------

def _unescape_json_pointer_token(token: str) -> str:
    """Unescape a single JSON Pointer reference token per RFC 6901."""
    # ~1 -> /  must be done before ~0 -> ~
    return token.replace("~1", "/").replace("~0", "~")


def resolve_json_pointer(document: dict, pointer: str) -> Any:
    """
    Resolve an RFC 6901 JSON Pointer against a document.

    Parameters
    ----------
    document : dict
        The root document to resolve against. Typically the full
        ``zarr.json`` content of a node, e.g.
        ``{"zarr_format": 3, "node_type": "array", "attributes": {...}}``.
    pointer : str
        A fully-qualified JSON Pointer starting with ``/``, e.g.
        ``"/attributes/crs/WGS84"``.

    Returns
    -------
    Any
        The value at the pointed-to location.

    Raises
    ------
    XGroupPathError
        If the pointer is malformed or the path does not exist in the
        document.
    """
    if not pointer.startswith("/"):
        raise XGroupPathError(
            source_path="<unknown>",
            attribute="attribute",
            target_path=pointer,
            reason=_("JSON Pointer must start with '/'."),
        )

    # Empty pointer refers to the whole document
    if pointer == "/":
        return document

    tokens = [_unescape_json_pointer_token(t) for t in pointer[1:].split("/")]
    current = document
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                raise XGroupPathError(
                    source_path="<unknown>",
                    attribute="attribute",
                    target_path=pointer,
                    reason=_("Key '{token}' not found.").format(token=token),
                )
            current = current[token]
        elif isinstance(current, list):
            try:
                idx = int(token)
            except ValueError:
                raise XGroupPathError(
                    source_path="<unknown>",
                    attribute="attribute",
                    target_path=pointer,
                    reason=_(
                        "Cannot index list with non-integer token '{token}'."
                    ).format(token=token),
                )
            try:
                current = current[idx]
            except IndexError:
                raise XGroupPathError(
                    source_path="<unknown>",
                    attribute="attribute",
                    target_path=pointer,
                    reason=_(
                        "List index {idx} out of range."
                    ).format(idx=idx),
                )
        else:
            raise XGroupPathError(
                source_path="<unknown>",
                attribute="attribute",
                target_path=pointer,
                reason=_(
                    "Cannot descend into scalar value at token '{token}'."
                ).format(token=token),
            )
    return current


# ---------------------------------------------------------------------------
# Node path resolution
# ---------------------------------------------------------------------------

_URI_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://")


def _is_uri(s: str) -> bool:
    """Return True if ``s`` looks like an absolute URI with a scheme."""
    return bool(_URI_RE.match(s))


def _resolve_relative_path(base_path: str, relative: str) -> str:
    """
    Resolve a relative node path against a base group path.

    Parameters
    ----------
    base_path : str
        Absolute path of the referencing node's group, e.g. ``"/ocean"``.
        Use ``"/"`` or ``""`` for the root group.
    relative : str
        Relative path from the base group, e.g. ``"../grid/lon"``.

    Returns
    -------
    str
        Resolved absolute path, e.g. ``"/grid/lon"``.
    """
    # Normalise base to a list of non-empty segments
    base_parts = [p for p in base_path.strip("/").split("/") if p]

    rel_parts = relative.split("/")
    result = base_parts[:]

    for part in rel_parts:
        if part == "..":
            if result:
                result.pop()
            # at root — stay at root
        elif part == "." or part == "":
            pass
        else:
            result.append(part)

    return "/" + "/".join(result) if result else "/"


def _get_node(store_root: zarr.Group, abs_path: str, source_path: str, attribute: str):
    """
    Retrieve a zarr node (Array or Group) at an absolute path within a store.

    Parameters
    ----------
    store_root : zarr.Group
        Root group of the store.
    abs_path : str
        Absolute path within the store, starting with ``/``.
    source_path : str
        Path of the declaring array, for error messages.
    attribute : str
        Attribute name containing the reference, for error messages.

    Returns
    -------
    zarr.Array or zarr.Group
    """
    # zarr-python uses paths without leading slash
    key = abs_path.lstrip("/")
    try:
        if key == "" or key == "/":
            return store_root
        return store_root[key]
    except KeyError:
        raise XGroupReferenceError(
            source_path=source_path,
            attribute=attribute,
            target_path=abs_path,
            reason=_("Node not found in store."),
        )
    except Exception as exc:
        raise XGroupReferenceError(
            source_path=source_path,
            attribute=attribute,
            target_path=abs_path,
            reason=str(exc),
        )


def _open_external_store(
    uri: str,
    source_path: str,
    cross_store_storage_options: dict | None,
) -> zarr.Group:
    """
    Open an external Zarr store by URI.

    Parameters
    ----------
    uri : str
        URI of the external store.
    source_path : str
        Path of the declaring array, for error messages.
    cross_store_storage_options : dict or None
        Mapping of URI prefix to storage options dict. The longest
        matching prefix is used.

    Returns
    -------
    zarr.Group
        Root group of the external store.
    """
    storage_options = {}
    if cross_store_storage_options:
        # find the longest matching prefix
        match = max(
            (prefix for prefix in cross_store_storage_options if uri.startswith(prefix)),
            key=len,
            default=None,
        )
        if match:
            storage_options = cross_store_storage_options[match]

    try:
        return zarr.open(uri, mode="r", storage_options=storage_options or None)
    except Exception as exc:
        raise XGroupStoreError(
            source_path=source_path,
            store_uri=uri,
            reason=str(exc),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_ref(
    ref: dict,
    *,
    referencing_group: zarr.Group,
    store_root: zarr.Group,
    source_path: str = "<unknown>",
    attribute: str = "ref",
    cross_store_storage_options: dict | None = None,
) -> Any:
    """
    Resolve a ``ref`` object to a zarr node or a metadata value.

    Parameters
    ----------
    ref : dict
        A ``ref`` object with ``node`` (required), and optionally
        ``uri`` and ``attribute`` fields.
    referencing_group : zarr.Group
        The group containing the referencing array. Used as the base
        for relative path resolution.
    store_root : zarr.Group
        Root group of the current store.
    source_path : str
        Absolute path of the declaring array. Used in error messages.
    attribute : str
        Name of the attribute containing this reference. Used in error
        messages.
    cross_store_storage_options : dict or None
        Mapping of URI prefix to zarr storage options, for cross-store
        references.

    Returns
    -------
    zarr.Array, zarr.Group, or Any
        - If ``attribute`` is absent: the target ``zarr.Array`` or
          ``zarr.Group``.
        - If ``attribute`` is present: the Python value at the JSON
          Pointer location within the target node's attributes.

    Raises
    ------
    XGroupPathError
        If ``node`` is missing from the ref object, the path is
        malformed, or a JSON Pointer cannot be resolved.
    XGroupReferenceError
        If the target node does not exist in the store.
    XGroupStoreError
        If a cross-store URI cannot be opened.
    """
    if not isinstance(ref, dict):
        raise XGroupPathError(
            source_path=source_path,
            attribute=attribute,
            target_path=repr(ref),
            reason=_("ref value must be a JSON object."),
        )

    node_path = ref.get("node")
    if node_path is None:
        raise XGroupPathError(
            source_path=source_path,
            attribute=attribute,
            target_path="<missing>",
            reason=_("ref object is missing required 'node' field."),
        )

    uri = ref.get("uri")
    json_pointer = ref.get("attribute")

    # ------------------------------------------------------------------
    # 1. Determine the target store and resolve node path to absolute
    # ------------------------------------------------------------------
    if uri is not None:
        # Cross-store reference: node_path is absolute within the
        # external store
        target_store = _open_external_store(
            uri, source_path, cross_store_storage_options
        )
        abs_path = "/" + node_path.lstrip("/")
    elif node_path.startswith("/"):
        # Absolute path within current store
        target_store = store_root
        abs_path = node_path
    else:
        # Relative path from referencing group
        target_store = store_root
        base_path = referencing_group.path or "/"
        abs_path = _resolve_relative_path(base_path, node_path)

    # ------------------------------------------------------------------
    # 2. Fetch the target node
    # ------------------------------------------------------------------
    node = _get_node(target_store, abs_path, source_path, attribute)

    # ------------------------------------------------------------------
    # 3. Apply JSON Pointer if present
    # ------------------------------------------------------------------
    if json_pointer is None:
        return node

    # Build the full zarr.json document for the node so the pointer
    # can traverse zarr_format, node_type, attributes, etc.
    try:
        node_attrs = dict(node.attrs)
    except Exception as exc:
        raise XGroupReferenceError(
            source_path=source_path,
            attribute=attribute,
            target_path=abs_path,
            reason=_("Cannot read attributes: {exc}").format(exc=exc),
        )

    # Reconstruct a minimal zarr.json-like document for pointer traversal
    zarr_doc: dict[str, Any] = {"attributes": node_attrs}
    try:
        zarr_doc["zarr_format"] = node.metadata.zarr_format
        zarr_doc["node_type"] = (
            "array" if isinstance(node, zarr.Array) else "group"
        )
    except Exception:
        pass

    return resolve_json_pointer(zarr_doc, json_pointer)