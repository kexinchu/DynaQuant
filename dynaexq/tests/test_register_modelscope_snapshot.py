from __future__ import annotations

import pytest

from scripts.register_modelscope_snapshot import (
    content_set_sha256,
    normalize_catalog,
    verify_manifest_against_catalog,
)


def _raw_catalog():
    return [
        {
            "Type": "blob",
            "Path": "weights.safetensors",
            "Size": 10,
            "Sha256": "b" * 64,
            "Revision": "1" * 40,
            "CommittedDate": 10,
        },
        {
            "Type": "blob",
            "Path": "config.json",
            "Size": 5,
            "Sha256": "a" * 64,
            "Revision": "2" * 40,
            "CommittedDate": 11,
        },
    ]


def test_modelscope_content_set_is_order_invariant():
    first = normalize_catalog(_raw_catalog())
    second = normalize_catalog(list(reversed(_raw_catalog())))
    assert first == second
    assert content_set_sha256(first) == content_set_sha256(second)
    assert len(content_set_sha256(first)) == 64


def test_modelscope_catalog_verifies_complete_local_manifest():
    catalog = normalize_catalog(_raw_catalog())
    manifest = {
        "files": [
            {
                "path": record["path"],
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
            for record in catalog
        ]
    }
    verify_manifest_against_catalog(manifest, catalog)
    manifest["files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="mismatched"):
        verify_manifest_against_catalog(manifest, catalog)


def test_modelscope_catalog_rejects_invalid_digest():
    raw = _raw_catalog()
    raw[0]["Sha256"] = "short"
    with pytest.raises(ValueError, match="invalid file record"):
        normalize_catalog(raw)
