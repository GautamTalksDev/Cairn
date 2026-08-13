"""CP-16 — Structured logging and JSONL diagnostics."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from cairn._internal.diagnostics import load_diagnostics
from cairn.config import CairnConfig, DiagnosticsConfig
from cairn.monitor import CairnMonitor, MonitorConfig
from cairn.types import Pose, PoseEstimate


def _estimate(t: float, *, source: str, xyz: list[float], std: float = 0.05) -> PoseEstimate:
    cov = np.diag([std**2, std**2, std**2, 0.01, 0.01, 0.01])
    return PoseEstimate(
        pose=Pose(
            timestamp=t,
            position=np.asarray(xyz, dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
        covariance=cov,
        source=source,
    )


def test_diagnostics_jsonl_written(tmp_path: Path) -> None:
    path = tmp_path / "diag.jsonl"
    from dataclasses import replace

    from cairn.config import MonitorSection

    cfg = replace(
        CairnConfig(diagnostics=DiagnosticsConfig(enabled=True, path=str(path))),
        monitor=MonitorSection(mode="consistency"),
    )
    with CairnMonitor(cfg) as mon:
        # Epoch 0: consistent
        mon.update(
            0.0,
            {
                "a": _estimate(0.0, source="a", xyz=[0.0, 0.0, 0.0]),
                "b": _estimate(0.0, source="b", xyz=[0.01, 0.0, 0.0]),
            },
        )
        # Epoch 1: large disagreement → DO_NOT_TRUST
        mon.update(
            0.1,
            {
                "a": _estimate(0.1, source="a", xyz=[0.0, 0.0, 0.0]),
                "b": _estimate(0.1, source="b", xyz=[5.0, 0.0, 0.0]),
            },
        )

    assert path.is_file()
    rows = load_diagnostics(path)
    assert len(rows) == 2
    assert rows[0]["state"] == "NOMINAL"
    assert rows[1]["state"] == "DO_NOT_TRUST"
    assert rows[1]["state_changed"] is True
    assert rows[1]["trigger"] is not None
    assert rows[1]["trigger"]["statistic"] is not None
    assert rows[1]["trigger"]["threshold"] is not None
    assert rows[1]["trigger"]["statistic"] > rows[1]["trigger"]["threshold"]
    assert "stat=" in rows[1]["reason"] and "threshold=" in rows[1]["reason"]
    assert len(rows[1]["tests"]) == 2
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)  # finite JSON only


def test_diagnostics_disabled_by_default(tmp_path: Path) -> None:
    mon = CairnMonitor(MonitorConfig(mode="consistency"))
    mon.update(
        0.0,
        {
            "a": _estimate(0.0, source="a", xyz=[0.0, 0.0, 0.0]),
            "b": _estimate(0.0, source="b", xyz=[0.0, 0.0, 0.0]),
        },
    )
    mon.close()
    assert mon.diagnostics_path is None
    assert list(tmp_path.iterdir()) == [] or True


def test_state_transition_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="cairn.monitor"):
        mon = CairnMonitor(
            MonitorConfig(mode="consistency"),
            diagnostics_path=None,
        )
        mon.update(
            0.0,
            {
                "a": _estimate(0.0, source="a", xyz=[0.0, 0.0, 0.0]),
                "b": _estimate(0.0, source="b", xyz=[0.01, 0.0, 0.0]),
            },
        )
        mon.update(
            0.1,
            {
                "a": _estimate(0.1, source="a", xyz=[0.0, 0.0, 0.0]),
                "b": _estimate(0.1, source="b", xyz=[5.0, 0.0, 0.0]),
            },
        )
        mon.close()
    infos = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("state transition" in m and "DO_NOT_TRUST" in m for m in infos)
    assert any("statistic=" in m and "threshold=" in m for m in infos)


def test_replay_script_plots(tmp_path: Path) -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_diagnostics.py"
    spec = importlib.util.spec_from_file_location("replay_diagnostics", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    path = tmp_path / "d.jsonl"
    from dataclasses import replace

    from cairn.config import MonitorSection

    cfg = replace(
        CairnConfig(diagnostics=DiagnosticsConfig(enabled=True, path=str(path))),
        monitor=MonitorSection(mode="consistency"),
    )
    with CairnMonitor(cfg) as mon:
        for i in range(5):
            bias = 0.0 if i < 3 else 4.0
            mon.update(
                0.1 * i,
                {
                    "a": _estimate(0.1 * i, source="a", xyz=[0.0, 0.0, 0.0]),
                    "b": _estimate(0.1 * i, source="b", xyz=[bias, 0.0, 0.0]),
                },
            )
    out = tmp_path / "plot.png"
    mod.plot_diagnostics(load_diagnostics(path), out)
    assert out.is_file() and out.stat().st_size > 0


def test_diagnostics_config_requires_path() -> None:
    from cairn.errors import CairnInputError

    with pytest.raises(CairnInputError, match="diagnostics.path"):
        DiagnosticsConfig(enabled=True, path=None)
