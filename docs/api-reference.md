# API reference

Public surface only (`from cairn import ...`). Everything under `cairn._internal`
is unsupported.

Units and frames, unless a parameter says otherwise:

- Time: seconds (`float`)
- Position: metres `(x, y, z)`
- Orientation: unit quaternion **w-first** `(w, x, y, z)`
- Covariance: 6×6, order `[x, y, z, roll, pitch, yaw]` (m² / rad²)
- Protection levels: metres
- All sources in one `update` share one navigation frame


## Exceptions

### `CairnInputError`

Subclass of `ValueError`. Raised for malformed config or hard input failures the
monitor refuses to swallow (for example non-finite timestamps that break the
stream contract). Some filthy poses degrade to `DO_NOT_TRUST` instead of raising;
see `CairnMonitor.update`.


## Configuration

### `CairnConfig`

Frozen dataclass tree of all operational thresholds. Construct with defaults,
`CairnConfig.from_yaml(path)`, or by nesting section dataclasses.

**Key methods**

| Method | Returns | Notes |
| --- | --- | --- |
| `CairnConfig()` | `CairnConfig` | Built-in defaults (same spirit as `configs/default.yaml`) |
| `CairnConfig.from_yaml(path)` | `CairnConfig` | Loads YAML; unknown fields raise `CairnInputError` |
| `config.to_yaml(path)` | `None` | Writes effective config |

**Raises:** `CairnInputError` on invalid probabilities, non-positive windows,
bad covariances, or `diagnostics.enabled` without a path.

```python
from cairn import CairnConfig

cfg = CairnConfig.from_yaml("configs/default.yaml")
print(cfg.monitor.mode, cfg.integrity.p_fa)
```

Section fields are documented in [configuration.md](configuration.md).


## Monitor

### `CairnMonitor`

Streaming integrity monitor.

```text
CairnMonitor(config: CairnConfig | MonitorConfig | str | Path, *, diagnostics_path: str | Path | None = None)
```

| Parameter | Unit / type | Meaning |
| --- | --- | --- |
| `config` | `CairnConfig`, legacy `MonitorConfig`, or path to YAML | Operating thresholds and mode |
| `diagnostics_path` | path or `None` | Optional JSONL sink override |

**Raises:** `CairnInputError` if config validation fails.

#### `update`

```text
update(timestamp: float, estimates: dict[str, PoseEstimate], *, degeneracy_scores: ndarray | None = None) -> IntegrityReport
```

| Parameter | Unit / type | Meaning |
| --- | --- | --- |
| `timestamp` | s | Epoch time; must not go backwards |
| `estimates` | `dict[str, PoseEstimate]` | Source name → estimate in one shared frame |
| `degeneracy_scores` | optional length-3 array | Localizability scores for degeneracy mode |

**Returns:** `IntegrityReport` for this epoch.

**Raises:** `CairnInputError` on illegal timestamps or hard validation failures.

```python
from cairn import CairnConfig, CairnMonitor, Pose, PoseEstimate
import numpy as np

mon = CairnMonitor(CairnConfig())
q = np.array([1.0, 0.0, 0.0, 0.0])
cov = np.diag([0.05**2, 0.05**2, 0.05**2, 0.01, 0.01, 0.01])
pe = PoseEstimate(Pose(0.0, np.zeros(3), q), cov, "lidar")
report = mon.update(0.0, {"lidar": pe, "vio": pe})
print(report.state, report.protection_level_horizontal)
mon.close()
```

#### `close` / context manager

Flush diagnostics. Prefer `with CairnMonitor(...) as mon:`.

#### `reset`

Clear the sliding window and monitor state.

#### `state` (property)

Returns `MonitorState` snapshot of the stream (mode, last timestamp, window
occupancy). Useful for debugging, not a substitute for `IntegrityReport`.

#### `report_sequence`

Batch helper that feeds a time-ordered sequence of estimate dicts and returns
the list of reports. Prefer `update` in live code.


## Types

### `Pose`

```text
Pose(timestamp: float, position: array(3,), orientation: array(4,), frame: str | None = None)
```

| Field | Unit | Notes |
| --- | --- | --- |
| `timestamp` | s | Finite |
| `position` | m | Shape `(3,)` |
| `orientation` | quaternion | W-first; normalized on construct |
| `frame` | optional str | Reference frame name |

**Raises:** `ValueError` on bad shapes or non-finite values.

### `PoseEstimate`

```text
PoseEstimate(pose: Pose, covariance: array(6,6), source: str)
```

| Field | Unit | Notes |
| --- | --- | --- |
| `pose` | `Pose` | Timed pose |
| `covariance` | m² / rad² | SPD 6×6, `[x,y,z,roll,pitch,yaw]` |
| `source` | str | Non-empty provenance tag |

**Raises:** `ValueError` on non-SPD or non-finite covariance, empty source.

### `Trajectory`

Ordered `PoseEstimate` sequence with strictly increasing timestamps.

| Method | Returns | Notes |
| --- | --- | --- |
| `timestamps()` | `ndarray` | Seconds |
| `interpolate_at(t)` | `Pose` | Linear position, SLERP orientation |

**Raises:** `ValueError` if empty, unsorted, duplicate timestamps, or `t` out of
range for interpolation.

### `IntegrityState`

Enum: `NOMINAL`, `DEGRADED`, `DO_NOT_TRUST`.

### `IntegrityReport`

| Field | Unit | Meaning |
| --- | --- | --- |
| `timestamp` | s | Epoch time |
| `protection_level_horizontal` | m | HPL |
| `protection_level_vertical` | m | VPL |
| `state` | `IntegrityState` | Discrete integrity class |
| `reason` | str | Why this state |
| `time_since_corroboration` | s | Age of last cross-check when tracked |


## Metrics

All integrity metrics take reports paired with **horizontal** true errors in
metres (same length).

### `missed_detection_rate(reports, true_errors) -> float`

Fraction of epochs with `true_error > HPL` while `state == NOMINAL` (in `[0, 1]`).

**Raises:** `ValueError` on length mismatch or negative errors.

### `false_alarm_rate(reports, true_errors) -> float`

Fraction with non-nominal state while `true_error ≤ HPL`.

### `availability(reports) -> float`

Fraction with `state == NOMINAL`.

### `stanford_diagram(reports, true_errors, alert_limit) -> StanfordDiagramData`

Quadrant counts for the Stanford diagram. `alert_limit` in metres.

### `integrity_report_summary(reports, true_errors, alert_limit) -> IntegrityReportSummary`

Bundles MDR, FAR, availability, Stanford counts, and related headline fields.

### `position_errors(est, gt, align_mode="none", *, max_dt=None, with_scale=False) -> PositionErrorResult`

Per-match Euclidean, horizontal, and vertical errors (metres).

### `ate_rmse(est, gt, align_mode="umeyama", *, max_dt=None, with_scale=False) -> float`

Absolute trajectory error RMSE (metres) after optional alignment.

### `rte(est, gt, align_mode="none", *, window=1.0, max_dt=None, with_scale=False, window_tol=None) -> ndarray`

Relative trajectory error samples (metres) over the given window (seconds).

```python
from cairn import missed_detection_rate

mdr = missed_detection_rate(reports, true_errors_m)
print(f"MDR={mdr:.4f}")
```


## Version

`cairn.__version__` is a string (for example `"0.1.0"`).
