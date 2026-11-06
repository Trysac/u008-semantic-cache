"""Behavior tests for the public API of semantic_cache."""

from __future__ import annotations

import pytest

import semantic_cache


def test_normalize_trims_and_lowercases() -> None:
    assert semantic_cache.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        semantic_cache.normalize(value)


def test_version_is_semver() -> None:
    parts = semantic_cache.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
