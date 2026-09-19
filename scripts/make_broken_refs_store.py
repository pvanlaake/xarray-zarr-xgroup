"""
Generate broken_refs.zarr for xarray-zarr-xgroup tests.
Run from project root: python scripts/make_broken_refs_store.py

All ref paths are relative to the referencing array (not its group).
"""
import numpy as np
import zarr
import os, shutil
from pathlib import Path

STORE = Path(__file__).parent.parent / "zarr_xgroup" / "tests" / "stores" / "broken_refs.zarr"
if STORE.exists(): shutil.rmtree(STORE)
STORE.parent.mkdir(parents=True, exist_ok=True)

CS_UUID  = "e4dbf0b7-7a00-4ce6-b23e-484292014ab4"
REF_UUID = "d89b30cf-ed8c-43d5-9a16-b492f0cd8786"

root = zarr.open(str(STORE), mode="w")
root.attrs.update({"title": "broken references test store"})

g_bad = root.require_group("bad")
rng = np.random.default_rng(42)

# Array with reference to non-existent node
# From /bad/var: ../../nonexistent/coords → /nonexistent/coords (does not exist)
arr_bad = g_bad.create_array("var",
    data=rng.random((5, 5)).astype("f4"),
    chunks=(5, 5), dimension_names=["y", "x"])
arr_bad.attrs.update({
    "zarr_conventions": [
        {"uuid": CS_UUID,  "name": "cs"},
        {"uuid": REF_UUID, "name": "ref"},
    ],
    "cs": {
        "crs": [{
            "type": "planar",
            "axes": {
                "x": {
                    "abbreviation": "X",
                    "coordinates": [{
                        "direction": "east", "unit": "m",
                        "values": {"external": {"ref": {"node": "../../nonexistent/coords"}}}
                    }]
                },
                "y": {
                    "abbreviation": "Y",
                    "coordinates": [{
                        "direction": "north", "unit": "m",
                        "values": {"regular": [0.0, 1000.0]}
                    }]
                }
            }
        }]
    }
})

# Array with malformed ref (missing node field)
arr_malformed = g_bad.create_array("malformed",
    data=rng.random((5, 5)).astype("f4"),
    chunks=(5, 5), dimension_names=["y", "x"])
arr_malformed.attrs.update({
    "zarr_conventions": [
        {"uuid": CS_UUID,  "name": "cs"},
        {"uuid": REF_UUID, "name": "ref"},
    ],
    "cs": {
        "crs": [{
            "type": "planar",
            "axes": {
                "x": {
                    "abbreviation": "X",
                    "coordinates": [{
                        "direction": "east", "unit": "m",
                        "values": {"external": {"ref": {}}}  # missing "node"
                    }]
                }
            }
        }]
    }
})

print(f"Written {STORE}")
