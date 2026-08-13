"""Cairn: integrity monitoring for multi-source robot navigation.

Public API
----------
Integrators should use only the names in :data:`__all__`::

    from cairn import CairnMonitor, CairnConfig, PoseEstimate, IntegrityReport

Everything under ``cairn._internal`` is unsupported implementation detail.
"""

from __future__ import annotations

from cairn.config import CairnConfig
from cairn.errors import CairnInputError
from cairn.metrics import (
    IntegrityReportSummary,
    PositionErrorResult,
    StanfordDiagramData,
    ate_rmse,
    availability,
    false_alarm_rate,
    integrity_report_summary,
    missed_detection_rate,
    position_errors,
    rte,
    stanford_diagram,
)
from cairn.monitor import CairnMonitor, MonitorState
from cairn.types import (
    IntegrityReport,
    IntegrityState,
    Pose,
    PoseEstimate,
    Trajectory,
)

__version__ = "0.1.0"

__all__ = [
    "CairnConfig",
    "CairnInputError",
    "CairnMonitor",
    "IntegrityReport",
    "IntegrityReportSummary",
    "IntegrityState",
    "MonitorState",
    "Pose",
    "PoseEstimate",
    "PositionErrorResult",
    "StanfordDiagramData",
    "Trajectory",
    "__version__",
    "ate_rmse",
    "availability",
    "false_alarm_rate",
    "integrity_report_summary",
    "missed_detection_rate",
    "position_errors",
    "rte",
    "stanford_diagram",
]
