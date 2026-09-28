from __future__ import annotations

from pathlib import Path

import pytest

from hgcl.studies.s003.environment import (
    EnvironmentError,
    EnvironmentSnapshot,
    RequiredEnvironment,
    validate_environment,
)


def _snapshot(**changes: object) -> EnvironmentSnapshot:
    values = {
        "python_version": "3.11.15",
        "dependencies": {
            "torch": "2.6.0",
            "torch-geometric": "2.6.1",
            "scikit-learn": "1.6.1",
            "numpy": "2.4.6",
            "polars": "1.44.2",
            "pyarrow": "21.0.0",
            "PyYAML": "6.0.3",
            "scipy": "1.17.1",
            "xgboost": "3.2.0",
        },
        "device": "cuda",
        "cuda_available": True,
        "ram_gib": 32.0,
        "vram_gib": 6.0,
        "free_disk_gib": 100.0,
    }
    values.update(changes)
    return EnvironmentSnapshot(**values)


def _required(tmp_path: Path) -> RequiredEnvironment:
    return RequiredEnvironment(
        profile="lab",
        device="cuda",
        exact_dependencies=_snapshot().dependencies,
        max_ram_gib=24.0,
        max_vram_gib=4.5,
        min_free_disk_gib=20.0,
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts" / "s003",
    )


def test_compatible_environment_passes(tmp_path: Path) -> None:
    report = validate_environment(_required(tmp_path), _snapshot())
    assert report.ready
    assert report.violations == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"python_version": "3.12.1"},
        {"device": "cpu"},
        {"cuda_available": False},
        {"ram_gib": 20.0},
        {"vram_gib": 4.0},
        {"free_disk_gib": 10.0},
        {"dependencies": {**_snapshot().dependencies, "torch": "2.7.0"}},
    ],
)
def test_mismatch_fails_closed(tmp_path: Path, changes: dict[str, object]) -> None:
    with pytest.raises(EnvironmentError):
        validate_environment(_required(tmp_path), _snapshot(**changes))


def test_overlapping_paths_fail_closed(tmp_path: Path) -> None:
    required = _required(tmp_path)
    required = RequiredEnvironment(
        **{**required.__dict__, "artifacts_root": required.data_root / "artifacts"}
    )
    with pytest.raises(EnvironmentError, match="overlap"):
        validate_environment(required, _snapshot())
