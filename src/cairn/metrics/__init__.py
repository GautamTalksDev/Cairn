"""Public metrics API."""

from cairn.metrics.alignment import align_trajectories, associate, umeyama_alignment
from cairn.metrics.error import PositionErrorResult, ate_rmse, position_errors, rte
from cairn.metrics.integrity import (
    IntegrityReportSummary,
    StanfordDiagramData,
    availability,
    false_alarm_rate,
    integrity_report_summary,
    missed_detection_rate,
    stanford_diagram,
)

__all__ = [
    "associate",
    "umeyama_alignment",
    "align_trajectories",
    "PositionErrorResult",
    "position_errors",
    "ate_rmse",
    "rte",
    "missed_detection_rate",
    "false_alarm_rate",
    "availability",
    "stanford_diagram",
    "StanfordDiagramData",
    "integrity_report_summary",
    "IntegrityReportSummary",
]
