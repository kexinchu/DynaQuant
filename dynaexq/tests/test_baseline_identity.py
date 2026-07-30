from __future__ import annotations

import importlib

import pytest

from dynaexq.baselines.lru_offload import LRUOffloadCache


def test_blocking_lru_has_unambiguous_identity():
    assert LRUOffloadCache.__name__ == "LRUOffloadCache"


def test_expertflow_name_fails_closed_without_full_implementation():
    with pytest.raises(ImportError, match="full ExpertFlow implementation"):
        importlib.import_module("dynaexq.baselines.expertflow")
