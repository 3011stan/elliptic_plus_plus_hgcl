"""Shared fixtures restricted to the S003 test namespace."""

from __future__ import annotations

from pathlib import Path
import json

import pytest


S003_TEST_ROOT = Path(__file__).resolve().parent
S003_FIXTURE_ROOT = S003_TEST_ROOT / "fixtures"


@pytest.fixture(scope="session")
def s003_fixture_root() -> Path:
    """Return the fixture root without consulting historical-study fixtures."""
    return S003_FIXTURE_ROOT


@pytest.fixture(scope="session")
def s003_csv_paths(s003_fixture_root: Path) -> dict[str, Path]:
    """Return only the three core transaction fixture paths."""
    return {
        name: s003_fixture_root / name
        for name in ("txs_features.csv", "txs_classes.csv", "txs_edgelist.csv")
    }


@pytest.fixture(scope="session")
def s003_expected(s003_fixture_root: Path) -> dict:
    """Return deterministic counts and hashes for the tiny fixture."""
    return json.loads((s003_fixture_root / "expected.json").read_text(encoding="utf-8"))
