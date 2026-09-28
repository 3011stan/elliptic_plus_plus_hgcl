from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hgcl.studies.s003.data import DataError, discover_source, read_source


def test_core_source_schema_and_mapping(s003_fixture_root: Path) -> None:
    source = discover_source(s003_fixture_root)
    assert set(source.core_files) == {"txs_features.csv", "txs_classes.csv", "txs_edgelist.csv"}
    assert source.optional_files == {}
    loaded = read_source(source)
    assert len(loaded.feature_names) == 182
    assert len(loaded.source_columns) - 1 == 183
    assert loaded.labels == {"1001": 1, "1002": 0, "1003": -1, "1004": 1}


def test_optional_address_files_are_inventory_only(s003_fixture_root: Path) -> None:
    optional = s003_fixture_root / "addr_features.csv"
    optional.write_text("address,feature\na,1\n", encoding="utf-8")
    try:
        source = discover_source(s003_fixture_root)
        assert "addr_features.csv" in source.optional_files
    finally:
        optional.unlink()


def test_duplicate_ids_and_invalid_edge_references_fail(tmp_path: Path, s003_fixture_root: Path) -> None:
    for name in ("txs_features.csv", "txs_classes.csv", "txs_edgelist.csv"):
        (tmp_path / name).write_bytes((s003_fixture_root / name).read_bytes())
    with (tmp_path / "txs_features.csv").open("a", encoding="utf-8") as stream:
        stream.write((s003_fixture_root / "txs_features.csv").read_text().splitlines()[1] + "\n")
    with pytest.raises(DataError, match="duplicate"):
        read_source(discover_source(tmp_path))

    rows = list(csv.reader((s003_fixture_root / "txs_features.csv").open()))
    with (tmp_path / "txs_features.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    (tmp_path / "txs_edgelist.csv").write_text("txId1,txId2\n1001,missing\n", encoding="utf-8")
    with pytest.raises(DataError, match="missing"):
        read_source(discover_source(tmp_path))

