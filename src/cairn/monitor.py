"""Online streaming integrity monitor (public product API).

``CairnMonitor`` is the integration boundary: push one epoch of pose estimates
in a shared navigation frame and receive one :class:`~cairn.types.IntegrityReport`.
History is a **bounded** sliding window so memory does not grow with mission length.

Units / frames
--------------
* Timestamps — seconds (float).
* Positions — metres; orientations — w-first unit quaternions.
* Covariances — 6×6 ``[x,y,z,roll,pitch,yaw]`` (m² / rad²).
* Protection levels — metres.
* All sources in an ``update()`` call must share the same navigation frame.

Raises
------
CairnInputError
    Non-finite / backwards timestamps, invalid source names, or filthy poses
    that fail hard validation (recoverable filthy inputs degrade to
    ``DO_NOT_TRUST`` without crashing).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cairn._internal.baselines.naive import NaiveCovarianceMonitor
from cairn._internal.detectors.consistency import (
    SolutionSeparationDetector,
    diagnostic_parts_from_ss,
)
from cairn._internal.detectors.degeneracy import DegeneracyAwareMonitor
from cairn._internal.diagnostics import (
    DiagnosticWriter,
    TestStatRecord,
    TriggerRecord,
    diagnostic_from_report,
    log_epoch_debug,
    log_state_transition,
)
from cairn._internal.validate import untrusted_report, validate_pose_estimate
from cairn.config import CairnConfig, DiagnosticsConfig
from cairn.errors import CairnInputError
from cairn.types import IntegrityReport, IntegrityState, PoseEstimate, Trajectory

__all__ = [
    "CairnMonitor",
    "MonitorConfig",
    "MonitorMode",
    "MonitorState",
    "batch_consistency_reports",
    "batch_degeneracy_reports",
    "batch_naive_reports",
]

logger = logging.getLogger("cairn.monitor")

MonitorMode = Literal["naive", "consistency", "degeneracy"]

_DEFAULTS = CairnConfig()


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    """Legacy monitor-only knobs. Prefer :class:`~cairn.config.CairnConfig`."""

    mode: MonitorMode = _DEFAULTS.monitor.mode
    max_window_epochs: int = _DEFAULTS.window.max_epochs
    max_window_seconds: float = _DEFAULTS.window.max_seconds
    holdover_max_dt: float = _DEFAULTS.window.holdover_max_dt_s
    primary_source: str | None = _DEFAULTS.monitor.primary_source
    confidence: float = _DEFAULTS.integrity.chi2_confidence
    p_fa: float = _DEFAULTS.integrity.p_fa
    pl_confidence: float = _DEFAULTS.integrity.pl_confidence
    score_threshold: float = _DEFAULTS.degeneracy.score_threshold
    dnt_threshold: float = _DEFAULTS.degeneracy.dnt_threshold
    loc_floor: float = _DEFAULTS.degeneracy.loc_floor
    max_inflation: float = _DEFAULTS.degeneracy.max_inflation
    untrusted_pl_m: float = _DEFAULTS.integrity.untrusted_pl_m
    gap_warn_s: float = _DEFAULTS.association.gap_warn_s

    def __post_init__(self) -> None:
        _ = self.to_cairn_config()

    def to_cairn_config(self) -> CairnConfig:
        from cairn.config import (
            AssociationConfig,
            DegeneracyConfig,
            IntegrityThresholds,
            MonitorSection,
            WindowConfig,
        )

        return CairnConfig(
            integrity=IntegrityThresholds(
                chi2_confidence=self.confidence,
                p_fa=self.p_fa,
                pl_confidence=self.pl_confidence,
                untrusted_pl_m=self.untrusted_pl_m,
            ),
            window=WindowConfig(
                max_epochs=self.max_window_epochs,
                max_seconds=self.max_window_seconds,
                holdover_max_dt_s=self.holdover_max_dt,
            ),
            degeneracy=DegeneracyConfig(
                score_threshold=self.score_threshold,
                dnt_threshold=self.dnt_threshold,
                loc_floor=self.loc_floor,
                max_inflation=self.max_inflation,
            ),
            association=AssociationConfig(gap_warn_s=self.gap_warn_s),
            monitor=MonitorSection(mode=self.mode, primary_source=self.primary_source),
        )

    @classmethod
    def from_cairn_config(cls, cfg: CairnConfig) -> MonitorConfig:
        return cls(
            mode=cfg.monitor.mode,
            max_window_epochs=cfg.window.max_epochs,
            max_window_seconds=cfg.window.max_seconds,
            holdover_max_dt=cfg.window.holdover_max_dt_s,
            primary_source=cfg.monitor.primary_source,
            confidence=cfg.integrity.chi2_confidence,
            p_fa=cfg.integrity.p_fa,
            pl_confidence=cfg.integrity.pl_confidence,
            score_threshold=cfg.degeneracy.score_threshold,
            dnt_threshold=cfg.degeneracy.dnt_threshold,
            loc_floor=cfg.degeneracy.loc_floor,
            max_inflation=cfg.degeneracy.max_inflation,
            untrusted_pl_m=cfg.integrity.untrusted_pl_m,
            gap_warn_s=cfg.association.gap_warn_s,
        )


def _resolve_config(
    config: CairnConfig | MonitorConfig | str | Path,
) -> CairnConfig:
    if isinstance(config, CairnConfig):
        return config
    if isinstance(config, MonitorConfig):
        return config.to_cairn_config()
    return CairnConfig.from_yaml(config)


@dataclass(slots=True)
class _EpochRecord:
    timestamp: float
    estimates: dict[str, PoseEstimate]


@dataclass(frozen=True, slots=True)
class MonitorState:
    """Inspectable snapshot of :class:`CairnMonitor` internals."""

    mode: MonitorMode
    epoch_count: int
    window_epochs: int
    window_span_s: float
    last_timestamp: float | None
    known_sources: tuple[str, ...]
    last_report: IntegrityReport | None
    sources_present_last: tuple[str, ...]
    sources_held_over_last: tuple[str, ...]
    config: CairnConfig


class CairnMonitor:
    """Streaming integrity monitor with a bounded sliding history window.

    Parameters
    ----------
    config:
        :class:`~cairn.config.CairnConfig`, legacy :class:`MonitorConfig`, or
        path to a YAML config file.
    diagnostics_path:
        Optional JSONL path; enables per-epoch diagnostics when set.

    Raises
    ------
    CairnInputError
        If ``config`` / YAML is invalid.
    """

    def __init__(
        self,
        config: CairnConfig | MonitorConfig | str | Path,
        *,
        diagnostics_path: str | Path | None = None,
    ) -> None:
        self._config = _resolve_config(config)
        if diagnostics_path is not None:
            self._config = replace(
                self._config,
                diagnostics=DiagnosticsConfig(enabled=True, path=str(diagnostics_path)),
            )
        self._window: deque[_EpochRecord] = deque()
        self._epoch_count = 0
        self._last_timestamp: float | None = None
        self._last_report: IntegrityReport | None = None
        self._last_state: IntegrityState | None = None
        self._known_sources: set[str] = set()
        self._sources_present_last: tuple[str, ...] = ()
        self._sources_held_over_last: tuple[str, ...] = ()
        self._latest_by_source: dict[str, PoseEstimate] = {}

        cfg = self._config
        self._untrusted_pl = cfg.integrity.untrusted_pl_m
        self._naive = NaiveCovarianceMonitor(confidence=cfg.integrity.chi2_confidence)
        self._consistency = SolutionSeparationDetector(
            p_fa=cfg.integrity.p_fa,
            pl_confidence=cfg.integrity.pl_confidence,
            untrusted_pl_m=cfg.integrity.untrusted_pl_m,
        )
        self._degeneracy = DegeneracyAwareMonitor(
            confidence=cfg.integrity.chi2_confidence,
            score_threshold=cfg.degeneracy.score_threshold,
            dnt_threshold=cfg.degeneracy.dnt_threshold,
            loc_floor=cfg.degeneracy.loc_floor,
            max_inflation=cfg.degeneracy.max_inflation,
        )
        diag = cfg.diagnostics
        self._diag = DiagnosticWriter(diag.path, enabled=diag.enabled)

    @property
    def config(self) -> CairnConfig:
        return self._config

    @property
    def diagnostics_path(self) -> Path | None:
        return self._diag.path if self._diag.enabled else None

    @property
    def state(self) -> MonitorState:
        span = 0.0
        if self._window:
            span = float(self._window[-1].timestamp - self._window[0].timestamp)
        return MonitorState(
            mode=self._config.monitor.mode,
            epoch_count=self._epoch_count,
            window_epochs=len(self._window),
            window_span_s=span,
            last_timestamp=self._last_timestamp,
            known_sources=tuple(sorted(self._known_sources)),
            last_report=self._last_report,
            sources_present_last=self._sources_present_last,
            sources_held_over_last=self._sources_held_over_last,
            config=self._config,
        )

    def close(self) -> None:
        """Flush and close the diagnostic sink (idempotent)."""
        self._diag.close()

    def __enter__(self) -> CairnMonitor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def reset(self) -> None:
        """Clear window and counters; keep configuration and diagnostic sink."""
        self._window.clear()
        self._epoch_count = 0
        self._last_timestamp = None
        self._last_report = None
        self._last_state = None
        self._known_sources.clear()
        self._sources_present_last = ()
        self._sources_held_over_last = ()
        self._latest_by_source.clear()

    def update(
        self,
        timestamp: float,
        estimates: dict[str, PoseEstimate],
        *,
        degeneracy_scores: NDArray[np.float64] | None = None,
    ) -> IntegrityReport:
        """Ingest one epoch and return the integrity assessment.

        Parameters
        ----------
        timestamp:
            Epoch time in seconds (must be finite and non-decreasing).
        estimates:
            Map of source name → :class:`~cairn.types.PoseEstimate` in a shared
            navigation frame. Positions in metres; covariances in m²/rad².
        degeneracy_scores:
            Required when ``monitor.mode == "degeneracy"``: shape ``(6,)``
            localizability scores in ``[0, 1]`` for
            ``[x, y, z, roll, pitch, yaw]``.

        Returns
        -------
        IntegrityReport
            Protection levels in metres; ``state`` is NOMINAL / DEGRADED /
            DO_NOT_TRUST.

        Raises
        ------
        CairnInputError
            Backwards or non-finite ``timestamp``, or invalid ``estimates`` type /
            source names.
        """
        ts = float(timestamp)
        if not np.isfinite(ts):
            raise CairnInputError(f"timestamp must be finite, got {timestamp!r}")
        if self._last_timestamp is not None and ts + 1e-15 < self._last_timestamp:
            raise CairnInputError(
                f"timestamp went backwards: prev={self._last_timestamp:.9f}, "
                f"curr={ts:.9f}"
            )

        if not isinstance(estimates, dict):
            raise CairnInputError("estimates must be a dict[str, PoseEstimate]")

        present: dict[str, PoseEstimate] = {}
        for name, est in estimates.items():
            if not isinstance(name, str) or not name:
                raise CairnInputError(f"invalid source name {name!r}")
            try:
                validate_pose_estimate(est)
            except CairnInputError as exc:
                logger.warning("rejecting source %r: %s", name, exc)
                report = untrusted_report(
                    ts,
                    reason=f"input_rejected:{exc}",
                    source=name,
                    pl_m=self._untrusted_pl,
                )
                return self._finish_epoch(
                    ts,
                    estimates,
                    present=(),
                    held=(),
                    report=report,
                    threshold=None,
                    tests=(),
                    trigger=TriggerRecord(
                        name=name,
                        statistic=None,
                        threshold=None,
                        kind="input_rejected",
                        comparison="malformed_input",
                    ),
                    resolved_sources=(),
                )
            present[name] = est

        held: dict[str, PoseEstimate] = {}
        holdover = self._config.window.holdover_max_dt_s
        dropped: list[str] = []
        if holdover > 0.0:
            for name, latest in self._latest_by_source.items():
                if name in present:
                    continue
                age = ts - float(latest.pose.timestamp)
                if 0.0 <= age <= holdover + 1e-12:
                    held[name] = latest
                else:
                    dropped.append(name)
        if dropped:
            logger.warning(
                "dropped sources (hold-over expired) at t=%.9f: %s",
                ts,
                dropped,
            )

        resolved = {**held, **present}
        report, threshold, tests, trigger = self._assess(
            ts, resolved, degeneracy_scores=degeneracy_scores
        )
        return self._finish_epoch(
            ts,
            present,
            present=tuple(sorted(present)),
            held=tuple(sorted(held)),
            report=report,
            threshold=threshold,
            tests=tests,
            trigger=trigger,
            resolved_sources=tuple(sorted(resolved)),
        )

    def report_sequence(
        self,
        epochs: list[tuple[float, dict[str, PoseEstimate]]],
        *,
        degeneracy_scores: list[NDArray[np.float64] | None] | None = None,
    ) -> list[IntegrityReport]:
        """Reset and replay a recorded sequence through :meth:`update`."""
        self.reset()
        out: list[IntegrityReport] = []
        for i, (ts, ests) in enumerate(epochs):
            scores = degeneracy_scores[i] if degeneracy_scores is not None else None
            out.append(self.update(ts, ests, degeneracy_scores=scores))
        return out

    def _finish_epoch(
        self,
        ts: float,
        present_estimates: dict[str, PoseEstimate],
        *,
        present: tuple[str, ...],
        held: tuple[str, ...],
        report: IntegrityReport,
        threshold: float | None,
        tests: tuple[TestStatRecord, ...],
        trigger: TriggerRecord | None,
        resolved_sources: tuple[str, ...],
    ) -> IntegrityReport:
        prev = self._last_state
        log_state_transition(
            logger,
            timestamp=ts,
            prev=prev,
            new=report.state,
            reason=report.reason,
            trigger=trigger,
        )
        log_epoch_debug(
            logger,
            timestamp=ts,
            sources=resolved_sources,
            threshold=threshold,
            tests=tests,
            state=report.state,
            reason=report.reason,
        )
        self._diag.write(
            diagnostic_from_report(
                report,
                mode=self._config.monitor.mode,
                sources=resolved_sources,
                sources_held_over=held,
                threshold=threshold,
                tests=tests,
                trigger=trigger,
                prev_state=prev,
            )
        )

        self._window.append(
            _EpochRecord(timestamp=ts, estimates=dict(present_estimates))
        )
        self._prune_window(now=ts)
        self._epoch_count += 1
        self._last_timestamp = ts
        self._sources_present_last = present
        self._sources_held_over_last = held
        self._last_report = report
        self._last_state = report.state
        return report

    def _prune_window(self, *, now: float) -> None:
        cfg = self._config.window
        while self._window and (now - self._window[0].timestamp) > cfg.max_seconds + 1e-15:
            self._window.popleft()
        while len(self._window) > cfg.max_epochs:
            self._window.popleft()

        by_src: dict[str, PoseEstimate] = {}
        for rec in self._window:
            for name, est in rec.estimates.items():
                prev = by_src.get(name)
                if prev is None or float(est.pose.timestamp) >= float(prev.pose.timestamp):
                    by_src[name] = est
        self._latest_by_source = by_src
        self._known_sources = set(by_src.keys())

    def _primary(self, resolved: dict[str, PoseEstimate]) -> PoseEstimate | None:
        if not resolved:
            return None
        key = self._config.monitor.primary_source
        if key is not None:
            return resolved.get(key)
        if len(resolved) == 1:
            return next(iter(resolved.values()))
        return resolved[sorted(resolved.keys())[0]]

    def _assess(
        self,
        ts: float,
        resolved: dict[str, PoseEstimate],
        *,
        degeneracy_scores: NDArray[np.float64] | None,
    ) -> tuple[
        IntegrityReport,
        float | None,
        tuple[TestStatRecord, ...],
        TriggerRecord | None,
    ]:
        mode = self._config.monitor.mode
        if not resolved:
            logger.warning("skipped epoch assessment: no sources at t=%.9f", ts)
            report = untrusted_report(
                ts, reason="no_sources_at_epoch", pl_m=self._untrusted_pl
            )
            return (
                report,
                None,
                (),
                TriggerRecord(
                    name="__none__",
                    statistic=None,
                    threshold=None,
                    kind="skipped",
                    comparison="no_sources",
                ),
            )

        if mode == "naive":
            primary = self._primary(resolved)
            if primary is None:
                report = untrusted_report(
                    ts,
                    reason=(
                        f"primary_source={self._config.monitor.primary_source!r} "
                        "missing from resolved estimates"
                    ),
                    pl_m=self._untrusted_pl,
                )
                return report, None, (), None
            report = self._naive.report_one(primary)
            tests: tuple[TestStatRecord, ...] = (
                TestStatRecord(
                    name=primary.source or "primary",
                    statistic=float(report.protection_level_horizontal),
                    threshold=None,
                    crossed=False,
                    kind="naive_pl",
                ),
            )
            trigger = None
            if report.state is not IntegrityState.NOMINAL:
                trigger = TriggerRecord(
                    name=primary.source or "primary",
                    statistic=None,
                    threshold=None,
                    kind="naive_input",
                    comparison="malformed_or_untrusted",
                )
            return report, None, tests, trigger

        if mode == "consistency":
            result = self._consistency.detect(resolved)
            report = self._consistency.report_from_result(result, resolved)
            threshold, tests, trigger = diagnostic_parts_from_ss(result)
            return report, threshold, tests, trigger

        if mode == "degeneracy":
            primary = self._primary(resolved)
            if primary is None:
                report = untrusted_report(
                    ts, reason="degeneracy_missing_primary", pl_m=self._untrusted_pl
                )
                return report, None, (), None
            if degeneracy_scores is None:
                logger.warning(
                    "degeneracy scores missing at t=%.9f; DO_NOT_TRUST", ts
                )
                report = untrusted_report(
                    ts,
                    reason="degeneracy_scores_required",
                    source=primary.source,
                    pl_m=self._untrusted_pl,
                )
                return report, None, (), None
            report = self._degeneracy.report_one(primary, degeneracy_scores)
            threshold, tests, trigger = self._degeneracy.diagnostic_parts(
                degeneracy_scores, report
            )
            return report, threshold, tests, trigger

        raise RuntimeError(f"unhandled mode {mode!r}")


def batch_naive_reports(
    estimates: list[PoseEstimate],
    *,
    confidence: float | None = None,
) -> list[IntegrityReport]:
    """Batch naive reports for parity tests."""
    conf = (
        float(confidence)
        if confidence is not None
        else CairnConfig().integrity.chi2_confidence
    )
    return NaiveCovarianceMonitor(confidence=conf).report(Trajectory(estimates))


def batch_consistency_reports(
    epochs: list[dict[str, PoseEstimate]],
    *,
    p_fa: float | None = None,
    pl_confidence: float | None = None,
) -> list[IntegrityReport]:
    """Batch solution-separation reports for parity tests."""
    cfg = CairnConfig()
    det = SolutionSeparationDetector(
        p_fa=cfg.integrity.p_fa if p_fa is None else p_fa,
        pl_confidence=cfg.integrity.pl_confidence if pl_confidence is None else pl_confidence,
        untrusted_pl_m=cfg.integrity.untrusted_pl_m,
    )
    return [det.report_one(src) for src in epochs]


def batch_degeneracy_reports(
    estimates: list[PoseEstimate],
    scores: list[NDArray[np.float64]],
    *,
    confidence: float | None = None,
    score_threshold: float | None = None,
    dnt_threshold: float | None = None,
) -> list[IntegrityReport]:
    """Batch degeneracy reports for parity tests."""
    cfg = CairnConfig()
    mon = DegeneracyAwareMonitor(
        confidence=cfg.integrity.chi2_confidence if confidence is None else confidence,
        score_threshold=(
            cfg.degeneracy.score_threshold if score_threshold is None else score_threshold
        ),
        dnt_threshold=cfg.degeneracy.dnt_threshold if dnt_threshold is None else dnt_threshold,
        loc_floor=cfg.degeneracy.loc_floor,
        max_inflation=cfg.degeneracy.max_inflation,
    )
    return [mon.report_one(est, sc) for est, sc in zip(estimates, scores, strict=True)]
