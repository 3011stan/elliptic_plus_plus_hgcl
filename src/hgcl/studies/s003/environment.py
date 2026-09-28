"""Fail-closed environment and resource checks for S003."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
import platform
import shutil
from typing import Mapping


class EnvironmentError(RuntimeError):
    """Raised when the requested execution profile is not safe to run."""


@dataclass(frozen=True)
class EnvironmentSnapshot:
    python_version: str
    dependencies: Mapping[str, str]
    device: str
    cuda_available: bool
    ram_gib: float
    vram_gib: float
    free_disk_gib: float


@dataclass(frozen=True)
class RequiredEnvironment:
    profile: str
    device: str
    exact_dependencies: Mapping[str, str]
    max_ram_gib: float
    max_vram_gib: float
    min_free_disk_gib: float
    data_root: Path
    artifacts_root: Path


@dataclass(frozen=True)
class EnvironmentReport:
    ready: bool
    violations: tuple[str, ...]


def _paths_overlap(first: Path, second: Path) -> bool:
    left = first.resolve()
    right = second.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def validate_environment(
    required: RequiredEnvironment, snapshot: EnvironmentSnapshot
) -> EnvironmentReport:
    violations: list[str] = []
    if tuple(int(part) for part in snapshot.python_version.split(".")[:2]) != (3, 11):
        violations.append(f"Python 3.11 required, found {snapshot.python_version}")
    for package, expected in required.exact_dependencies.items():
        actual = snapshot.dependencies.get(package)
        if actual != expected:
            violations.append(f"dependency {package}: expected {expected}, found {actual}")
    if snapshot.device != required.device:
        violations.append(f"device: expected {required.device}, found {snapshot.device}")
    if required.device == "cuda" and not snapshot.cuda_available:
        violations.append("CUDA requested but unavailable")
    if snapshot.ram_gib < required.max_ram_gib:
        violations.append("available RAM is below the configured maximum requirement")
    if snapshot.vram_gib < required.max_vram_gib:
        violations.append("available VRAM is below the configured maximum requirement")
    if snapshot.free_disk_gib < required.min_free_disk_gib:
        violations.append("free disk is below the configured minimum")
    if _paths_overlap(required.data_root, required.artifacts_root):
        violations.append("data and artifact paths overlap")
    if violations:
        raise EnvironmentError("; ".join(violations))
    return EnvironmentReport(ready=True, violations=())


def probe_environment(required: RequiredEnvironment) -> EnvironmentSnapshot:
    dependencies: dict[str, str] = {}
    for package in required.exact_dependencies:
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            dependencies[package] = "missing"

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device = "cuda"
            vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            vram_gib = 0.0
        else:
            device = "cpu"
            vram_gib = 0.0
    except ImportError:
        cuda_available = False
        device = "cpu"
        vram_gib = 0.0

    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    ram_gib = page_size * page_count / 2**30
    free_disk_gib = shutil.disk_usage(required.artifacts_root.parent).free / 2**30
    return EnvironmentSnapshot(
        python_version=platform.python_version(),
        dependencies=dependencies,
        device=device,
        cuda_available=cuda_available,
        ram_gib=ram_gib,
        vram_gib=vram_gib,
        free_disk_gib=free_disk_gib,
    )


def doctor(required: RequiredEnvironment) -> EnvironmentReport:
    return validate_environment(required, probe_environment(required))
