"""CP-14 — Streaming API parity, bounded memory, and source dropout."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from cairn._internal.validate import CairnInputError
from cairn.monitor import (
    CairnMonitor,
    MonitorConfig,
    batch_consistency_reports,
    batch_degeneracy_reports,
    batch_naive_reports,
)
from cairn.types import IntegrityState, Pose, PoseEstimate


def _pose(t: float, xyz: list[float] | None = None) -> Pose:
    return Pose(
        timestamp=t,
        position=np.asarray(xyz if xyz is not None else [0.0, 0.0, 0.0], dtype=np.float64),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


def _estimate(
    t: float,
    *,
    source: str,
    xyz: list[float] | None = None,
    std: float = 0.05,
) -> PoseEstimate:
    cov = np.diag([std**2, std**2, std**2, 0.01, 0.01, 0.01])
    return PoseEstimate(pose=_pose(t, xyz), covariance=cov, source=source)


def _assert_reports_close(
    stream: list,
    batch: list,
    *,
    atol: float = 1e-9,
) -> None:
    assert len(stream) == len(batch)
    for i, (a, b) in enumerate(zip(stream, batch, strict=True)):
        assert a.state == b.state, f"epoch {i}: state {a.state} != {b.state}"
        assert a.timestamp == pytest.approx(b.timestamp, abs=atol)
        assert a.protection_level_horizontal == pytest.approx(
            b.protection_level_horizontal, abs=atol, rel=0.0
        )
        assert a.protection_level_vertical == pytest.approx(
            b.protection_level_vertical, abs=atol, rel=0.0
        )


# ---------------------------------------------------------------------------
# Streaming ↔ batch numerical agreement
# ---------------------------------------------------------------------------


class TestStreamingBatchParity:
    def test_naive_agrees_to_1e_9(self) -> None:
        n = 80
        estimates = [
            _estimate(0.1 * i, source="primary", xyz=[0.01 * i, 0.0, 0.0])
            for i in range(n)
        ]
        batch = batch_naive_reports(estimates, confidence=0.999)

        mon = CairnMonitor(
            MonitorConfig(
                mode="naive",
                primary_source="primary",
                max_window_epochs=20,
                max_window_seconds=5.0,
                holdover_max_dt=0.0,
                confidence=0.999,
            )
        )
        stream = mon.report_sequence(
            [(float(e.pose.timestamp), {"primary": e}) for e in estimates]
        )
        _assert_reports_close(stream, batch, atol=1e-9)

    def test_consistency_agrees_to_1e_9(self) -> None:
        n = 60
        epochs: list[dict[str, PoseEstimate]] = []
        for i in range(n):
            t = 0.1 * i
            epochs.append(
                {
                    "vio": _estimate(t, source="vio", xyz=[0.0, 0.0, 0.0], std=0.04),
                    "lidar": _estimate(
                        t, source="lidar", xyz=[0.02 * np.sin(i / 7), 0.0, 0.0], std=0.05
                    ),
                    "prop": _estimate(
                        t, source="prop", xyz=[0.01, 0.01 * np.cos(i / 5), 0.0], std=0.06
                    ),
                }
            )
        batch = batch_consistency_reports(epochs, p_fa=0.05, pl_confidence=0.999)

        mon = CairnMonitor(
            MonitorConfig(
                mode="consistency",
                max_window_epochs=30,
                max_window_seconds=10.0,
                holdover_max_dt=0.0,  # self-contained epochs
                p_fa=0.05,
                pl_confidence=0.999,
            )
        )
        stream = mon.report_sequence([(0.1 * i, epochs[i]) for i in range(n)])
        _assert_reports_close(stream, batch, atol=1e-9)

    def test_degeneracy_agrees_to_1e_9(self) -> None:
        n = 40
        estimates = [
            _estimate(0.2 * i, source="slam", xyz=[0.0, 0.05 * i, 0.0])
            for i in range(n)
        ]
        rng = np.random.default_rng(0)
        scores = [rng.uniform(0.05, 1.0, size=6) for _ in range(n)]
        batch = batch_degeneracy_reports(
            estimates, scores, confidence=0.999, score_threshold=0.25, dnt_threshold=0.08
        )

        mon = CairnMonitor(
            MonitorConfig(
                mode="degeneracy",
                primary_source="slam",
                max_window_epochs=15,
                max_window_seconds=8.0,
                holdover_max_dt=0.0,
                confidence=0.999,
                score_threshold=0.25,
                dnt_threshold=0.08,
            )
        )
        stream = mon.report_sequence(
            [(float(e.pose.timestamp), {"slam": e}) for e in estimates],
            degeneracy_scores=scores,
        )
        _assert_reports_close(stream, batch, atol=1e-9)


# ---------------------------------------------------------------------------
# Bounded memory
# ---------------------------------------------------------------------------


class TestBoundedMemory:
    def test_window_length_capped_by_epochs(self) -> None:
        max_epochs = 25
        mon = CairnMonitor(
            MonitorConfig(
                mode="naive",
                primary_source="primary",
                max_window_epochs=max_epochs,
                max_window_seconds=1e6,
                holdover_max_dt=0.0,
            )
        )
        lengths: list[int] = []
        n = 10 * max_epochs
        for i in range(n):
            est = _estimate(0.1 * i, source="primary")
            mon.update(0.1 * i, {"primary": est})
            lengths.append(mon.state.window_epochs)
            assert mon.state.window_epochs <= max_epochs
        # After fill, length must stay flat at the cap.
        assert lengths[-1] == max_epochs
        assert all(L == max_epochs for L in lengths[max_epochs:])

    def test_window_pruned_by_seconds(self) -> None:
        mon = CairnMonitor(
            MonitorConfig(
                mode="naive",
                primary_source="primary",
                max_window_epochs=10_000,
                max_window_seconds=1.0,
                holdover_max_dt=0.0,
            )
        )
        for i in range(50):
            t = 0.1 * i  # 5 s total
            mon.update(t, {"primary": _estimate(t, source="primary")})
            assert mon.state.window_span_s <= 1.0 + 1e-9
            assert mon.state.window_epochs <= 11  # 1.0 / 0.1 + 1

    def test_peak_memory_flat_across_10x_window(self) -> None:
        max_epochs = 40
        n = 10 * max_epochs
        mon = CairnMonitor(
            MonitorConfig(
                mode="consistency",
                max_window_epochs=max_epochs,
                max_window_seconds=1e6,
                holdover_max_dt=0.0,
                p_fa=0.05,
            )
        )

        tracemalloc.start()
        peaks: list[int] = []
        for i in range(n):
            t = 0.05 * i
            sources = {
                "a": _estimate(t, source="a", xyz=[0.0, 0.0, 0.0]),
                "b": _estimate(t, source="b", xyz=[0.01, 0.0, 0.0]),
            }
            mon.update(t, sources)
            assert mon.state.window_epochs <= max_epochs
            if i % max_epochs == max_epochs - 1:
                peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

        assert len(peaks) >= 3
        # Peak should not grow roughly linearly with mission length.
        # Allow noise but reject a clear upward trend (>2× from first filled window).
        assert peaks[-1] < 2.0 * peaks[0] + 1_000_000


# ---------------------------------------------------------------------------
# Source dropout / multi-rate
# ---------------------------------------------------------------------------


class TestSourceDropout:
    def test_dropout_degrades_without_crash(self) -> None:
        mon = CairnMonitor(
            MonitorConfig(
                mode="consistency",
                max_window_epochs=50,
                max_window_seconds=10.0,
                holdover_max_dt=0.0,  # no hold-over: absence is immediate
                p_fa=0.05,
            )
        )
        # Three sources present.
        t0 = 0.0
        r0 = mon.update(
            t0,
            {
                "a": _estimate(t0, source="a"),
                "b": _estimate(t0, source="b", xyz=[0.01, 0.0, 0.0]),
                "c": _estimate(t0, source="c", xyz=[-0.01, 0.0, 0.0]),
            },
        )
        assert r0.state in (IntegrityState.NOMINAL, IntegrityState.DO_NOT_TRUST)
        assert np.isfinite(r0.protection_level_horizontal)

        # Mid-stream: only one source remains.
        t1 = 0.1
        r1 = mon.update(t1, {"a": _estimate(t1, source="a")})
        assert r1.state == IntegrityState.DO_NOT_TRUST
        assert np.isfinite(r1.protection_level_horizontal)
        assert "single_source" in r1.reason
        assert mon.state.sources_present_last == ("a",)
        assert mon.state.sources_held_over_last == ()

    def test_holdover_bridges_multi_rate(self) -> None:
        mon = CairnMonitor(
            MonitorConfig(
                mode="consistency",
                max_window_epochs=100,
                max_window_seconds=5.0,
                holdover_max_dt=0.25,
                p_fa=0.05,
            )
        )
        # Slow source at t=0; fast source every 0.1 s.
        mon.update(
            0.0,
            {
                "slow": _estimate(0.0, source="slow", xyz=[0.0, 0.0, 0.0]),
                "fast": _estimate(0.0, source="fast", xyz=[0.01, 0.0, 0.0]),
            },
        )
        r = mon.update(0.1, {"fast": _estimate(0.1, source="fast", xyz=[0.01, 0.0, 0.0])})
        assert "slow" in mon.state.sources_held_over_last
        assert "fast" in mon.state.sources_present_last
        # With hold-over, still two sources → not forced single-source DNT.
        assert "single_source" not in r.reason
        assert np.isfinite(r.protection_level_horizontal)

    def test_holdover_expires(self) -> None:
        mon = CairnMonitor(
            MonitorConfig(
                mode="consistency",
                max_window_epochs=100,
                max_window_seconds=5.0,
                holdover_max_dt=0.15,
                p_fa=0.05,
            )
        )
        mon.update(
            0.0,
            {
                "slow": _estimate(0.0, source="slow"),
                "fast": _estimate(0.0, source="fast", xyz=[0.01, 0.0, 0.0]),
            },
        )
        # 0.2 s later > holdover_max_dt: slow expired.
        r = mon.update(0.2, {"fast": _estimate(0.2, source="fast", xyz=[0.01, 0.0, 0.0])})
        assert mon.state.sources_held_over_last == ()
        assert r.state == IntegrityState.DO_NOT_TRUST


class TestStateAndReset:
    def test_reset_clears_window(self) -> None:
        mon = CairnMonitor(
            MonitorConfig(mode="naive", primary_source="p", max_window_epochs=10)
        )
        for i in range(5):
            mon.update(float(i), {"p": _estimate(float(i), source="p")})
        assert mon.state.epoch_count == 5
        assert mon.state.window_epochs == 5
        mon.reset()
        assert mon.state.epoch_count == 0
        assert mon.state.window_epochs == 0
        assert mon.state.last_report is None
        assert mon.state.known_sources == ()

    def test_backward_timestamp_raises(self) -> None:
        mon = CairnMonitor(MonitorConfig(mode="naive", primary_source="p"))
        mon.update(1.0, {"p": _estimate(1.0, source="p")})
        with pytest.raises(CairnInputError, match="backwards"):
            mon.update(0.5, {"p": _estimate(0.5, source="p")})
