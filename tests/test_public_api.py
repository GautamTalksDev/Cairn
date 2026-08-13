"""CP-18 — Public API surface smoke checks."""

from __future__ import annotations

import cairn


def test_public_all_imports() -> None:
    for name in cairn.__all__:
        assert hasattr(cairn, name), name


def test_integrator_entrypoints() -> None:
    from cairn import CairnConfig, CairnMonitor, Pose, PoseEstimate

    assert CairnMonitor is cairn.CairnMonitor
    assert CairnConfig is cairn.CairnConfig
    _ = Pose, PoseEstimate


def test_internal_not_reexported() -> None:
    assert not hasattr(cairn, "SolutionSeparationDetector")
    assert not hasattr(cairn, "NaiveCovarianceMonitor")
    assert not hasattr(cairn, "load_tum")
