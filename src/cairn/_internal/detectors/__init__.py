"""Integrity / geometry detectors."""

from cairn._internal.detectors.consistency import (
    SolutionSeparationDetector,
    SolutionSeparationResult,
    fuse_positions_wls,
    solution_separation,
)
from cairn._internal.detectors.degeneracy import (
    DegeneracyAwareMonitor,
    DegeneracyDetector,
    DegeneracyUpdate,
    inflate_covariance_anisotropic,
    localizability,
    point_to_plane_hessian,
)

__all__ = [
    "DegeneracyDetector",
    "DegeneracyUpdate",
    "DegeneracyAwareMonitor",
    "localizability",
    "point_to_plane_hessian",
    "inflate_covariance_anisotropic",
    "SolutionSeparationDetector",
    "SolutionSeparationResult",
    "solution_separation",
    "fuse_positions_wls",
]
