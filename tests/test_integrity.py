"""Hand-constructed integrity metric cases with answers fixed by inspection."""

from __future__ import annotations

import numpy as np
import pytest

from cairn.metrics.integrity import (
    availability,
    false_alarm_rate,
    integrity_report_summary,
    missed_detection_rate,
    stanford_diagram,
)
from cairn.types import IntegrityReport, IntegrityState


def _report(
    *,
    t: float,
    pl: float,
    state: IntegrityState,
    pl_v: float | None = None,
) -> IntegrityReport:
    return IntegrityReport(
        timestamp=t,
        protection_level_horizontal=pl,
        protection_level_vertical=pl if pl_v is None else pl_v,
        state=state,
        reason="test",
        time_since_corroboration=0.0,
    )


class TestMissedDetectionRate:
    def test_all_safe(self) -> None:
        # Expected MDR = 0.0: every epoch is NOMINAL with error <= PL.
        reports = [
            _report(t=0.0, pl=2.0, state=IntegrityState.NOMINAL),
            _report(t=1.0, pl=3.0, state=IntegrityState.NOMINAL),
            _report(t=2.0, pl=1.5, state=IntegrityState.NOMINAL),
        ]
        true_errors = [0.5, 1.0, 1.5]  # all <= corresponding PL
        assert missed_detection_rate(reports, true_errors) == 0.0

    def test_all_missed(self) -> None:
        # Expected MDR = 1.0: every epoch is NOMINAL yet error > PL
        # (confidently wrong at every sample).
        reports = [
            _report(t=0.0, pl=1.0, state=IntegrityState.NOMINAL),
            _report(t=1.0, pl=0.5, state=IntegrityState.NOMINAL),
        ]
        true_errors = [1.5, 2.0]
        assert missed_detection_rate(reports, true_errors) == 1.0

    def test_fifty_fifty_split(self) -> None:
        # Expected MDR = 0.5: two epochs; only the first is a missed detection
        # (NOMINAL and error > PL). The second is NOMINAL but safe (error <= PL).
        reports = [
            _report(t=0.0, pl=1.0, state=IntegrityState.NOMINAL),
            _report(t=1.0, pl=2.0, state=IntegrityState.NOMINAL),
        ]
        true_errors = [1.5, 1.0]
        assert missed_detection_rate(reports, true_errors) == pytest.approx(0.5)

    def test_empty_input(self) -> None:
        # Expected MDR = 0.0: no epochs ⇒ no missed detections (empty-safe pipe).
        assert missed_detection_rate([], []) == 0.0


class TestFalseAlarmRate:
    def test_all_safe_no_alarms(self) -> None:
        # Expected FAR = 0.0: NOMINAL throughout, even though alarms would be
        # false if raised (error <= PL).
        reports = [
            _report(t=0.0, pl=2.0, state=IntegrityState.NOMINAL),
            _report(t=1.0, pl=2.0, state=IntegrityState.NOMINAL),
        ]
        assert false_alarm_rate(reports, [0.1, 0.2]) == 0.0

    def test_all_false_alarms(self) -> None:
        # Expected FAR = 1.0: every epoch is DEGRADED/DO_NOT_TRUST while error <= PL.
        reports = [
            _report(t=0.0, pl=2.0, state=IntegrityState.DEGRADED),
            _report(t=1.0, pl=3.0, state=IntegrityState.DO_NOT_TRUST),
        ]
        true_errors = [0.5, 1.0]
        assert false_alarm_rate(reports, true_errors) == 1.0

    def test_fifty_fifty_split(self) -> None:
        # Expected FAR = 0.5: first epoch false-alarms (DEGRADED, error <= PL);
        # second is a justified alarm (DO_NOT_TRUST with error > PL) — not a FAR.
        reports = [
            _report(t=0.0, pl=2.0, state=IntegrityState.DEGRADED),
            _report(t=1.0, pl=1.0, state=IntegrityState.DO_NOT_TRUST),
        ]
        true_errors = [0.5, 2.0]
        assert false_alarm_rate(reports, true_errors) == pytest.approx(0.5)

    def test_empty_input(self) -> None:
        # Expected FAR = 0.0: no epochs ⇒ no false alarms.
        assert false_alarm_rate([], []) == 0.0


class TestAvailability:
    def test_all_nominal(self) -> None:
        # Expected availability = 1.0: all three epochs NOMINAL.
        reports = [
            _report(t=0.0, pl=1.0, state=IntegrityState.NOMINAL),
            _report(t=1.0, pl=1.0, state=IntegrityState.NOMINAL),
            _report(t=2.0, pl=1.0, state=IntegrityState.NOMINAL),
        ]
        assert availability(reports) == 1.0

    def test_fifty_fifty_split(self) -> None:
        # Expected availability = 0.5: one NOMINAL, one DEGRADED.
        reports = [
            _report(t=0.0, pl=1.0, state=IntegrityState.NOMINAL),
            _report(t=1.0, pl=1.0, state=IntegrityState.DEGRADED),
        ]
        assert availability(reports) == pytest.approx(0.5)

    def test_empty_input(self) -> None:
        # Expected availability = 0.0: no epochs are NOMINAL.
        assert availability([]) == 0.0


class TestStanfordDiagram:
    def test_all_safe_nominal_quadrant(self) -> None:
        # AL = 5. Expected: all 2 points in nominal (error <= PL < AL).
        # (err, PL) = (1, 2) and (2, 3) → both under the diagonal and under AL.
        reports = [
            _report(t=0.0, pl=2.0, state=IntegrityState.NOMINAL),
            _report(t=1.0, pl=3.0, state=IntegrityState.NOMINAL),
        ]
        diagram = stanford_diagram(reports, [1.0, 2.0], alert_limit=5.0)
        assert diagram.nominal_count == 2
        assert diagram.misleading_count == 0
        assert diagram.hazardously_misleading_count == 0
        assert diagram.unavailable_count == 0

    def test_one_of_each_quadrant(self) -> None:
        # AL = 4. Four hand-placed points, one per region:
        #   nominal:     err=1, PL=2  (err <= PL < AL)
        #   misleading:  err=3, PL=1  (PL < err <= AL)
        #   HMI:         err=5, PL=2  (err > AL and PL < AL)
        #   unavailable: err=1, PL=5  (PL >= AL)
        reports = [
            _report(t=0.0, pl=2.0, state=IntegrityState.NOMINAL),
            _report(t=1.0, pl=1.0, state=IntegrityState.NOMINAL),
            _report(t=2.0, pl=2.0, state=IntegrityState.NOMINAL),
            _report(t=3.0, pl=5.0, state=IntegrityState.DEGRADED),
        ]
        true_errors = [1.0, 3.0, 5.0, 1.0]
        diagram = stanford_diagram(reports, true_errors, alert_limit=4.0)
        assert diagram.nominal_count == 1
        assert diagram.misleading_count == 1
        assert diagram.hazardously_misleading_count == 1
        assert diagram.unavailable_count == 1
        np.testing.assert_array_equal(diagram.true_errors, true_errors)
        np.testing.assert_array_equal(diagram.protection_levels, [2.0, 1.0, 2.0, 5.0])

    def test_empty_input(self) -> None:
        # Expected: empty arrays and zero counts for every quadrant.
        diagram = stanford_diagram([], [], alert_limit=1.0)
        assert diagram.true_errors.size == 0
        assert diagram.protection_levels.size == 0
        assert diagram.nominal_count == 0
        assert diagram.misleading_count == 0
        assert diagram.hazardously_misleading_count == 0
        assert diagram.unavailable_count == 0


class TestIntegrityReportSummary:
    def test_all_missed_summary_pipe(self) -> None:
        # Expected MDR = 1.0, FAR = 0.0, availability = 1.0, epoch_count = 2.
        # Both epochs NOMINAL with error > PL; Stanford: both MI if AL is large.
        reports = [
            _report(t=0.0, pl=1.0, state=IntegrityState.NOMINAL),
            _report(t=1.0, pl=0.5, state=IntegrityState.NOMINAL),
        ]
        true_errors = [2.0, 3.0]
        summary = integrity_report_summary(reports, true_errors, alert_limit=10.0)
        assert summary.epoch_count == 2
        assert summary.missed_detection_rate == 1.0
        assert summary.false_alarm_rate == 0.0
        assert summary.availability == 1.0
        assert summary.stanford.misleading_count == 2
        # Current missed-detection rate as a single number (pipe check).
        assert summary.missed_detection_rate == pytest.approx(1.0)

    def test_empty_summary(self) -> None:
        # Expected: epoch_count 0, all rates 0.0, Stanford counts 0.
        summary = integrity_report_summary([], [], alert_limit=1.0)
        assert summary.epoch_count == 0
        assert summary.missed_detection_rate == 0.0
        assert summary.false_alarm_rate == 0.0
        assert summary.availability == 0.0
        assert summary.stanford.nominal_count == 0
