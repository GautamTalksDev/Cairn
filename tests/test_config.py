"""CP-15 — Configuration load / validate / dump."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cairn._internal.validate import DEFAULT_GAP_WARN_S, UNTRUSTED_PL_M
from cairn.config import CairnConfig, DiagonalCovarianceConfig, IntegrityThresholds
from cairn.errors import CairnInputError
from cairn.monitor import CairnMonitor


def test_default_yaml_loads(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "default.yaml"
    assert path.is_file()
    cfg = CairnConfig.from_yaml(path)
    assert cfg.integrity.p_fa == pytest.approx(0.05)
    assert cfg.window.max_epochs == 100
    assert cfg.timing.duty_threshold == pytest.approx(0.5)
    assert cfg.monitor.mode == "consistency"
    assert cfg.diagnostics.enabled is False


def test_round_trip_yaml(tmp_path: Path) -> None:
    cfg = CairnConfig()
    out = tmp_path / "eff.yaml"
    text = cfg.to_yaml(out)
    assert out.is_file()
    loaded = CairnConfig.from_yaml(out)
    assert loaded.to_dict() == cfg.to_dict()
    assert "p_fa" in text


def test_rejects_negative_probability() -> None:
    with pytest.raises(CairnInputError, match="chi2_confidence"):
        IntegrityThresholds(chi2_confidence=-0.1)


def test_rejects_zero_window() -> None:
    with pytest.raises(CairnInputError, match="window.max_seconds"):
        CairnConfig.from_mapping({"window": {"max_seconds": 0.0}})


def test_rejects_non_positive_definite_covariance() -> None:
    with pytest.raises(CairnInputError, match="std_xy_m"):
        DiagonalCovarianceConfig(std_xy_m=-1.0)


def test_unknown_field_named() -> None:
    with pytest.raises(CairnInputError, match="unknown field"):
        CairnConfig.from_mapping({"integrity": {"not_a_field": 1.0}})


def test_validate_aliases_match_config() -> None:
    cfg = CairnConfig()
    assert UNTRUSTED_PL_M == pytest.approx(cfg.integrity.untrusted_pl_m)
    assert DEFAULT_GAP_WARN_S == pytest.approx(cfg.association.gap_warn_s)


def test_monitor_accepts_yaml_path(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "monitor": {"mode": "naive", "primary_source": "p"},
                "integrity": {"chi2_confidence": 0.99, "p_fa": 0.1, "pl_confidence": 0.99},
            }
        ),
        encoding="utf-8",
    )
    mon = CairnMonitor(path)
    assert mon.config.monitor.mode == "naive"
    assert mon.config.integrity.p_fa == pytest.approx(0.1)


def test_covariance_for_per_source() -> None:
    cfg = CairnConfig.from_mapping(
        {
            "default_covariance": {"std_xy_m": 0.05, "std_z_m": 0.05, "std_rot_rad": 0.05},
            "per_source_covariance": {
                "lidar": {"std_xy_m": 0.1, "std_z_m": 0.1, "std_rot_rad": 0.05}
            },
        }
    )
    assert cfg.covariance_for("lidar").std_xy_m == pytest.approx(0.1)
    assert cfg.covariance_for("other").std_xy_m == pytest.approx(0.05)
