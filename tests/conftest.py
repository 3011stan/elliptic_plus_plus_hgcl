"""Shared test locations; original-data tests are explicitly marked."""
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
