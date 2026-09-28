from __future__ import annotations

import pytest

from hgcl.studies.s003.data import DataError, audit_feature_availability, feature_groups


def test_feature_groups_exclude_identifier_and_time(s003_csv_paths) -> None:
    header = s003_csv_paths["txs_features.csv"].read_text().splitlines()[0].split(",")
    groups = feature_groups(header)
    assert tuple(map(len, (groups.local, groups.aggregate, groups.augmented))) == (93, 72, 17)
    assert groups.excluded == ("txId", "Time step")
    assert set(groups.model).isdisjoint(groups.excluded)


def test_causal_audit_is_versioned_and_fails_closed(s003_csv_paths) -> None:
    header = s003_csv_paths["txs_features.csv"].read_text().splitlines()[0].split(",")
    evidence = {name: {"causal": True, "source": "dictionary-v1"} for name in ("local", "aggregate", "augmented")}
    audit = audit_feature_availability(header, evidence, schema_version=1)
    assert audit.passed and audit.schema_version == 1
    evidence["aggregate"] = {"causal": False, "source": "future aggregation"}
    with pytest.raises(DataError, match="aggregate"):
        audit_feature_availability(header, evidence, schema_version=1)

