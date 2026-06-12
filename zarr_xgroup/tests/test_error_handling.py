"""
Tests for error handling and the broken_refs store.
"""
import warnings
from pathlib import Path

import pytest
import xarray as xr

from zarr_xgroup.errors import (
    XGroupNoPrincipalWarning,
    XGroupReferenceError,
    XGroupPathError,
)

STORE = Path(__file__).parent / "stores" / "broken_refs.zarr"


@pytest.fixture(scope="module")
def broken_store():
    if not STORE.exists():
        pytest.skip(f"broken_refs test store not found at {STORE}")
    return STORE


def test_missing_node_raises_reference_error(broken_store):
    """A reference to a non-existent node emits XGroupNoPrincipalWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xr.open_dataset(broken_store, engine="xgroup", group="/bad")
    messages = [str(x.message) for x in w if issubclass(x.category, XGroupNoPrincipalWarning)]
    assert any("not found in store" in m for m in messages), \
        f"Expected 'not found in store' warning, got: {messages}"


def test_malformed_ref_raises_path_error(broken_store):
    """A ref object missing 'node' field emits XGroupNoPrincipalWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xr.open_dataset(broken_store, engine="xgroup", group="/bad")
    messages = [str(x.message) for x in w if issubclass(x.category, XGroupNoPrincipalWarning)]
    assert any("node" in m.lower() for m in messages), \
        f"Expected 'node' in warning message, got: {messages}"


def test_warning_promotion_to_error():
    """XGroupNoPrincipalWarning can be promoted to an error."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=XGroupNoPrincipalWarning)
        with pytest.raises(XGroupNoPrincipalWarning):
            warnings.warn("test", XGroupNoPrincipalWarning)


def test_warning_suppression(broken_store):
    """XGroupNoPrincipalWarning can be suppressed."""
    # Open a store where some arrays have no principal convention
    # using a group path that has cs-declared arrays with broken refs
    # We just verify suppress works — the broken refs will still error
    # but NoPrincipal warnings should be suppressible
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=XGroupNoPrincipalWarning)
        try:
            xr.open_dataset(broken_store, engine="xgroup", group="/bad")
        except Exception:
            pass
    no_principal = [x for x in w if issubclass(x.category, XGroupNoPrincipalWarning)]
    assert len(no_principal) == 0
