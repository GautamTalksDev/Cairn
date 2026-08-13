"""Structured diagnostics and JSONL epoch dumps (CP-16).

Opt-in via :class:`~cairn.config.DiagnosticsConfig`. When enabled, each
monitor epoch writes one JSON line so a run can be plotted offline without
re-executing detectors.
"""

from __future__ import annotations

__all__ = [
    "DiagnosticWriter",
    "EpochDiagnostic",
    "TestStatRecord",
    "TriggerRecord",
    "diagnostic_from_report",
    "load_diagnostics",
    "log_epoch_debug",
    "log_state_transition",
]


import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cairn.types import IntegrityReport, IntegrityState

logger = logging.getLogger("cairn.diagnostics")


@dataclass(frozen=True, slots=True)
class TestStatRecord:
    """One hypothesis / axis test for post-hoc analysis."""

    name: str
    statistic: float | None
    threshold: float | None
    crossed: bool
    kind: str = "test"  # e.g. separation | localizability | naive_pl

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statistic": _json_float(self.statistic),
            "threshold": _json_float(self.threshold),
            "crossed": self.crossed,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class TriggerRecord:
    """The statistic that drove a non-NOMINAL decision (if any)."""

    name: str
    statistic: float | None
    threshold: float | None
    kind: str
    comparison: str  # e.g. "stat > threshold" or "score < threshold"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statistic": _json_float(self.statistic),
            "threshold": _json_float(self.threshold),
            "kind": self.kind,
            "comparison": self.comparison,
        }


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not np.isfinite(v):
        return None
    return v


@dataclass(frozen=True, slots=True)
class EpochDiagnostic:
    """One epoch of integrity diagnostics (JSONL record)."""

    timestamp: float
    mode: str
    state: str
    reason: str
    protection_level_horizontal: float
    protection_level_vertical: float
    sources: tuple[str, ...] = ()
    sources_held_over: tuple[str, ...] = ()
    threshold: float | None = None
    tests: tuple[TestStatRecord, ...] = ()
    trigger: TriggerRecord | None = None
    prev_state: str | None = None
    state_changed: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "timestamp": float(self.timestamp),
            "mode": self.mode,
            "state": self.state,
            "reason": self.reason,
            "protection_level_horizontal": _json_float(self.protection_level_horizontal),
            "protection_level_vertical": _json_float(self.protection_level_vertical),
            "sources": list(self.sources),
            "sources_held_over": list(self.sources_held_over),
            "threshold": _json_float(self.threshold),
            "tests": [t.to_dict() for t in self.tests],
            "trigger": None if self.trigger is None else self.trigger.to_dict(),
            "prev_state": self.prev_state,
            "state_changed": self.state_changed,
        }
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


class DiagnosticWriter:
    """Append-only JSONL sink. No-op when disabled."""

    def __init__(self, path: str | Path | None, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled) and path is not None
        self.path = Path(path) if path is not None else None
        self._fh = None
        if self.enabled:
            assert self.path is not None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")
            logger.info("diagnostic JSONL enabled → %s", self.path)

    def write(self, epoch: EpochDiagnostic) -> None:
        if not self.enabled or self._fh is None:
            return
        self._fh.write(json.dumps(epoch.to_dict(), allow_nan=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> DiagnosticWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_diagnostics(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL diagnostics file into a list of dicts."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def log_state_transition(
    log: logging.Logger,
    *,
    timestamp: float,
    prev: IntegrityState | None,
    new: IntegrityState,
    reason: str,
    trigger: TriggerRecord | None = None,
) -> None:
    """INFO when integrity state changes; includes triggering statistic when known."""
    if prev is new:
        return
    prev_name = "None" if prev is None else prev.name
    if trigger is not None:
        log.info(
            "state transition %s → %s at t=%.9f reason=%s "
            "trigger=%s statistic=%s threshold=%s (%s)",
            prev_name,
            new.name,
            timestamp,
            reason,
            trigger.name,
            "n/a" if trigger.statistic is None else f"{trigger.statistic:.6g}",
            "n/a" if trigger.threshold is None else f"{trigger.threshold:.6g}",
            trigger.comparison,
        )
    else:
        log.info(
            "state transition %s → %s at t=%.9f reason=%s",
            prev_name,
            new.name,
            timestamp,
            reason,
        )


def log_epoch_debug(
    log: logging.Logger,
    *,
    timestamp: float,
    sources: Sequence[str],
    threshold: float | None,
    tests: Sequence[TestStatRecord],
    state: IntegrityState,
    reason: str,
) -> None:
    """DEBUG dump of per-epoch test statistics and sources."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    parts = []
    for t in tests:
        stat = "n/a" if t.statistic is None else f"{t.statistic:.6g}"
        thr = "n/a" if t.threshold is None else f"{t.threshold:.6g}"
        mark = ">" if t.crossed else "≤"
        parts.append(f"{t.name}:stat={stat}/thr={thr}{mark}")
    log.debug(
        "epoch t=%.9f state=%s sources=%s threshold=%s tests=[%s] reason=%s",
        timestamp,
        state.name,
        list(sources),
        f"{threshold:.6g}" if threshold is not None else "n/a",
        "; ".join(parts) if parts else "none",
        reason,
    )


def diagnostic_from_report(
    report: IntegrityReport,
    *,
    mode: str,
    sources: Sequence[str] = (),
    sources_held_over: Sequence[str] = (),
    threshold: float | None = None,
    tests: Sequence[TestStatRecord] = (),
    trigger: TriggerRecord | None = None,
    prev_state: IntegrityState | None = None,
    extra: Mapping[str, Any] | None = None,
) -> EpochDiagnostic:
    """Build an :class:`EpochDiagnostic` from a report plus optional test stats."""
    return EpochDiagnostic(
        timestamp=float(report.timestamp),
        mode=mode,
        state=report.state.name,
        reason=report.reason,
        protection_level_horizontal=float(report.protection_level_horizontal),
        protection_level_vertical=float(report.protection_level_vertical),
        sources=tuple(sources),
        sources_held_over=tuple(sources_held_over),
        threshold=threshold,
        tests=tuple(tests),
        trigger=trigger,
        prev_state=None if prev_state is None else prev_state.name,
        state_changed=prev_state is not None and prev_state is not report.state,
        extra=dict(extra or {}),
    )
