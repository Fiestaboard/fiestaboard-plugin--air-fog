"""Shared fixtures for the air_fog plugin's tests."""

import json
from pathlib import Path

import pytest

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "manifest.json"


@pytest.fixture(scope="session")
def manifest() -> dict:
    """The plugin's real manifest.json, parsed once per session."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)
