"""Partner-tunable configuration for Cairn (public).

All operational thresholds live here and in ``configs/default.yaml``.
Partners retune without editing source: load YAML → construct
:class:`~cairn.monitor.CairnMonitor`.

Units
-----
Probabilities are dimensionless in ``(0, 1)``. Lengths are metres. Times are
seconds. Covariance stds are metres (position) or radians (orientation).

Raises
------
CairnInputError
    Invalid probabilities, non-positive windows, non-SPD default covariances,
    unknown YAML fields, or ``diagnostics.enabled`` without a path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray

from cairn.errors import CairnInputError

__all__ = [
    "AlignMode",
    "AssociationConfig",
    "CairnConfig",
    "DegeneracyConfig",
    "DiagnosticsConfig",
    "DiagonalCovarianceConfig",
    "EvaluationConfig",
    "IntegrityThresholds",
    "MonitorMode",
    "MonitorSection",
    "TimingMetricConfig",
    "WindowConfig",
]

MonitorMode = Literal["naive", "consistency", "degeneracy"]
AlignMode = Literal["none", "origin", "umeyama"]


def _require_prob(name: str, value: float) -> float:
    v = float(value)
    if not 0.0 < v < 1.0:
        raise CairnInputError(f"{name} must be in (0, 1), got {value!r}")
    return v


def _require_positive(name: str, value: float) -> float:
    v = float(value)
    if v <= 0.0:
        raise CairnInputError(f"{name} must be > 0, got {value!r}")
    return v


def _require_nonneg(name: str, value: float) -> float:
    v = float(value)
    if v < 0.0:
        raise CairnInputError(f"{name} must be >= 0, got {value!r}")
    return v


@dataclass(frozen=True, slots=True)
class DiagonalCovarianceConfig:
    """Diagonal pose covariance defaults (metres / radians)."""

    std_xy_m: float = 0.05
    std_z_m: float = 0.05
    std_rot_rad: float = 0.05

    def __post_init__(self) -> None:
        _require_positive("std_xy_m", self.std_xy_m)
        _require_positive("std_z_m", self.std_z_m)
        _require_positive("std_rot_rad", self.std_rot_rad)
        # Constructing the diagonal proves it is SPD.
        cov = self.as_matrix()
        eig = np.linalg.eigvalsh(cov)
        if float(eig.min()) <= 0.0:
            raise CairnInputError(
                "default covariance is not positive-definite "
                f"(min eigenvalue={float(eig.min()):.3e}); "
                f"check std_xy_m/std_z_m/std_rot_rad"
            )

    def as_matrix(self) -> NDArray[np.float64]:
        d = np.array(
            [
                self.std_xy_m**2,
                self.std_xy_m**2,
                self.std_z_m**2,
                self.std_rot_rad**2,
                self.std_rot_rad**2,
                self.std_rot_rad**2,
            ],
            dtype=np.float64,
        )
        return np.diag(d)


@dataclass(frozen=True, slots=True)
class IntegrityThresholds:
    """Protection-level and alert thresholds."""

    chi2_confidence: float = 0.999
    p_fa: float = 0.05
    pl_confidence: float = 0.999
    alert_limit_m: float = 2.0
    untrusted_pl_m: float = 1.0e6

    def __post_init__(self) -> None:
        _require_prob("chi2_confidence", self.chi2_confidence)
        _require_prob("p_fa", self.p_fa)
        _require_prob("pl_confidence", self.pl_confidence)
        _require_positive("alert_limit_m", self.alert_limit_m)
        _require_positive("untrusted_pl_m", self.untrusted_pl_m)


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Streaming history window."""

    max_epochs: int = 100
    max_seconds: float = 10.0
    holdover_max_dt_s: float = 0.5

    def __post_init__(self) -> None:
        if int(self.max_epochs) < 1:
            raise CairnInputError(f"window.max_epochs must be >= 1, got {self.max_epochs!r}")
        object.__setattr__(self, "max_epochs", int(self.max_epochs))
        _require_positive("window.max_seconds", self.max_seconds)
        _require_nonneg("window.holdover_max_dt_s", self.holdover_max_dt_s)


@dataclass(frozen=True, slots=True)
class TimingMetricConfig:
    """Duty-cycle time-to-detection metric (CP-11)."""

    duty_window_s: float = 5.0
    duty_threshold: float = 0.50
    require_full_post_onset_window: bool = True

    def __post_init__(self) -> None:
        _require_positive("timing.duty_window_s", self.duty_window_s)
        v = float(self.duty_threshold)
        if not 0.0 < v <= 1.0:
            raise CairnInputError(
                f"timing.duty_threshold must be in (0, 1], got {self.duty_threshold!r}"
            )
        object.__setattr__(self, "duty_threshold", v)


@dataclass(frozen=True, slots=True)
class DegeneracyConfig:
    """Localizability / degeneracy monitor knobs."""

    score_threshold: float = 0.25
    dnt_threshold: float = 0.08
    loc_floor: float = 0.05
    max_inflation: float = 50.0

    def __post_init__(self) -> None:
        for name in ("score_threshold", "dnt_threshold", "loc_floor"):
            v = float(getattr(self, name))
            if not 0.0 < v <= 1.0:
                raise CairnInputError(f"degeneracy.{name} must be in (0, 1], got {v!r}")
        _require_positive("degeneracy.max_inflation", self.max_inflation)
        if self.dnt_threshold >= self.score_threshold:
            raise CairnInputError(
                "degeneracy.dnt_threshold must be < degeneracy.score_threshold "
                f"(got {self.dnt_threshold} >= {self.score_threshold})"
            )


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    """Timestamp association and gap warnings."""

    max_dt_s: float = 0.05
    gap_warn_s: float = 1.0
    alignment_max_dt_s: float = 0.02

    def __post_init__(self) -> None:
        _require_nonneg("association.max_dt_s", self.max_dt_s)
        _require_nonneg("association.gap_warn_s", self.gap_warn_s)
        _require_nonneg("association.alignment_max_dt_s", self.alignment_max_dt_s)


@dataclass(frozen=True, slots=True)
class MonitorSection:
    """Streaming monitor mode selection."""

    mode: MonitorMode = "consistency"
    primary_source: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("naive", "consistency", "degeneracy"):
            raise CairnInputError(f"monitor.mode must be naive|consistency|degeneracy, got {self.mode!r}")
        if self.primary_source is not None and not str(self.primary_source):
            raise CairnInputError("monitor.primary_source must be non-empty when set")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Offline scoring defaults."""

    align_mode: AlignMode = "umeyama"

    def __post_init__(self) -> None:
        if self.align_mode not in ("none", "origin", "umeyama"):
            raise CairnInputError(
                f"evaluation.align_mode must be none|origin|umeyama, got {self.align_mode!r}"
            )


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """Opt-in per-epoch JSONL diagnostics (verbose; off by default)."""

    enabled: bool = False
    path: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and (self.path is None or not str(self.path).strip()):
            raise CairnInputError(
                "diagnostics.path must be set when diagnostics.enabled is true"
            )
        if self.path is not None and not str(self.path).strip():
            raise CairnInputError("diagnostics.path must be non-empty when set")


@dataclass(frozen=True, slots=True)
class CairnConfig:
    """Full Cairn configuration — load from YAML, dump for reproducibility.

    Parameters
    ----------
    integrity:
        Chi-squared confidence, false-alarm probability, alert limit (m), etc.
    window:
        Streaming history bound (epochs / seconds) and hold-over (s).
    timing:
        Duty-cycle detection metric knobs (offline evaluation).
    degeneracy:
        Localizability score thresholds (dimensionless) and inflation caps.
    association:
        Timestamp match tolerances (s).
    default_covariance / per_source_covariance:
        Claimed diagonal σ when a source omits covariance (m / rad).
    monitor:
        Streaming backend (``naive`` | ``consistency`` | ``degeneracy``).
    evaluation:
        Offline alignment mode for true-error scoring.
    diagnostics:
        Opt-in JSONL epoch dump (verbose).

    Raises
    ------
    CairnInputError
        On construction / ``from_yaml`` / ``from_mapping`` if any field is invalid.
    """

    integrity: IntegrityThresholds = field(default_factory=IntegrityThresholds)
    window: WindowConfig = field(default_factory=WindowConfig)
    timing: TimingMetricConfig = field(default_factory=TimingMetricConfig)
    degeneracy: DegeneracyConfig = field(default_factory=DegeneracyConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    default_covariance: DiagonalCovarianceConfig = field(
        default_factory=DiagonalCovarianceConfig
    )
    per_source_covariance: Mapping[str, DiagonalCovarianceConfig] = field(
        default_factory=dict
    )
    monitor: MonitorSection = field(default_factory=MonitorSection)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    def __post_init__(self) -> None:
        # Freeze per-source map as plain dict of validated configs.
        raw = dict(self.per_source_covariance)
        fixed: dict[str, DiagonalCovarianceConfig] = {}
        for name, cfg in raw.items():
            if not name:
                raise CairnInputError("per_source_covariance keys must be non-empty")
            if isinstance(cfg, DiagonalCovarianceConfig):
                fixed[name] = cfg
            elif isinstance(cfg, Mapping):
                fixed[name] = DiagonalCovarianceConfig(**_subset(cfg, DiagonalCovarianceConfig))
            else:
                raise CairnInputError(
                    f"per_source_covariance[{name!r}] must be a mapping, got {type(cfg)}"
                )
        object.__setattr__(self, "per_source_covariance", fixed)

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CairnConfig:
        """Build from a nested mapping (e.g. parsed YAML)."""
        data = dict(data)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise CairnInputError(
                f"unknown config section(s): {sorted(unknown)}"
            )
        kwargs: dict[str, Any] = {}
        if "integrity" in data:
            kwargs["integrity"] = IntegrityThresholds(
                **_subset(data["integrity"], IntegrityThresholds)
            )
        if "window" in data:
            kwargs["window"] = WindowConfig(**_subset(data["window"], WindowConfig))
        if "timing" in data:
            kwargs["timing"] = TimingMetricConfig(**_subset(data["timing"], TimingMetricConfig))
        if "degeneracy" in data:
            kwargs["degeneracy"] = DegeneracyConfig(
                **_subset(data["degeneracy"], DegeneracyConfig)
            )
        if "association" in data:
            kwargs["association"] = AssociationConfig(
                **_subset(data["association"], AssociationConfig)
            )
        if "default_covariance" in data:
            kwargs["default_covariance"] = DiagonalCovarianceConfig(
                **_subset(data["default_covariance"], DiagonalCovarianceConfig)
            )
        if "per_source_covariance" in data:
            kwargs["per_source_covariance"] = data["per_source_covariance"]
        if "monitor" in data:
            kwargs["monitor"] = MonitorSection(**_subset(data["monitor"], MonitorSection))
        if "evaluation" in data:
            kwargs["evaluation"] = EvaluationConfig(
                **_subset(data["evaluation"], EvaluationConfig)
            )
        if "diagnostics" in data:
            kwargs["diagnostics"] = DiagnosticsConfig(
                **_subset(data["diagnostics"], DiagnosticsConfig)
            )
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CairnConfig:
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise CairnInputError(
                f"config root must be a mapping, got {type(data).__name__} in {path}"
            )
        try:
            return cls.from_mapping(data)
        except CairnInputError as exc:
            raise CairnInputError(f"{exc} (in {path})") from exc

    @classmethod
    def default_path(cls) -> Path:
        """Repo ``configs/default.yaml`` when present, else package-relative guess."""
        # Prefer CWD/repo layout used in development.
        candidates = [
            Path.cwd() / "configs" / "default.yaml",
            Path(__file__).resolve().parents[2] / "configs" / "default.yaml",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return candidates[0]

    @classmethod
    def load_default(cls) -> CairnConfig:
        path = cls.default_path()
        if path.is_file():
            return cls.from_yaml(path)
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict suitable for YAML/JSON dumps."""
        d = asdict(self)
        # asdict converts nested dataclasses; per_source is already a dict.
        return d

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Dump effective config. If ``path`` is set, also write the file."""
        text = yaml.safe_dump(
            self.to_dict(),
            sort_keys=False,
            default_flow_style=False,
        )
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def covariance_for(self, source: str | None = None) -> DiagonalCovarianceConfig:
        """Resolve per-source override or the global default."""
        if source and source in self.per_source_covariance:
            return self.per_source_covariance[source]
        return self.default_covariance

    def with_overrides(self, **sections: Any) -> CairnConfig:
        """Return a copy with selected top-level sections replaced."""
        return replace(self, **sections)


def _subset(data: Mapping[str, Any], cls: type) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise CairnInputError(
            f"unknown field(s) for {cls.__name__}: {sorted(unknown)}"
        )
    return {k: data[k] for k in data if k in allowed}
