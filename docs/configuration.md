# Configuration

All operational knobs live in `CairnConfig` and `configs/default.yaml`. Copy the
YAML, edit, and pass it to `CairnMonitor` or `cairn score --config`.

These tuning stories are **illustrative**, not field-validated. Read
[limitations.md](limitations.md) before you treat any setting as certified.

## Field reference

### `integrity`

| Field | Default | Unit | What it does | Raise | Lower |
| --- | --- | --- | --- | --- | --- |
| `chi2_confidence` | 0.999 | probability in (0,1) | Chi-squared confidence for naive / degeneracy PL | Larger PL, safer look, less useful | Tighter PL, more MDR risk |
| `p_fa` | 0.05 | probability in (0,1) | Family-wise false-alarm target for solution separation | More alarms, lower availability | Quieter, higher MDR risk |
| `pl_confidence` | 0.999 | probability in (0,1) | Confidence inside SS protection-level bound | Larger SS HPL/VPL | Tighter SS PL |
| `alert_limit_m` | 2.0 | m | Stanford / ops alert limit | Fewer HMI labels | Stricter ops limit |
| `untrusted_pl_m` | 1e6 | m | Finite PL when epoch is untrusted / uncomputable | Still huge | Only if you understand MDR math |

### `window`

| Field | Default | Unit | What it does | Raise | Lower |
| --- | --- | --- | --- | --- | --- |
| `max_epochs` | 100 | count | Streaming history length | Longer memory, more RAM | Forget sources sooner |
| `max_seconds` | 10.0 | s | Max age of retained epochs | Longer temporal context | Tighter real-time footprint |
| `holdover_max_dt_s` | 0.5 | s | Reuse a missing source if last sample younger than this | Tolerate slower sensors | Require every source each tick (0 = no holdover) |

### `timing`

Used by evaluation scripts for duty-cycle time-to-detection (not the live PL
formula itself).

| Field | Default | Unit | What it does | Raise | Lower |
| --- | --- | --- | --- | --- | --- |
| `duty_window_s` | 5.0 | s | Rolling window for alarm duty | Stabler, later detection | Faster, noisier |
| `duty_threshold` | 0.50 | fraction in (0,1] | Fraction of window that must be alarmed | Stricter / later Δt | Earlier detection |
| `require_full_post_onset_window` | true | bool | Only score duty once the full window is after onset | Blocks pre-fault FAR leakage into Δt | Allows early contaminated Δt |

### `degeneracy`

| Field | Default | Unit | What it does | Raise | Lower |
| --- | --- | --- | --- | --- | --- |
| `score_threshold` | 0.25 | 0-1 | Localizability below this → `DEGRADED` | More degraded flags | Only severe under-constraint |
| `dnt_threshold` | 0.08 | 0-1 | Below this → `DO_NOT_TRUST` (must be &lt; `score_threshold`) | More DNT | Reserve DNT for near-zero information |
| `loc_floor` | 0.05 | 0-1 | Floor on localizability when inflating covariance | Less aggressive inflation | Larger PL in weak directions |
| `max_inflation` | 50.0 | factor | Cap on per-axis std inflation | Allow huge corridor PL | Limit PL blow-up |

### `association`

| Field | Default | Unit | What it does | Raise | Lower |
| --- | --- | --- | --- | --- | --- |
| `max_dt_s` | 0.05 | s | Match report times to GT / multi-rate sources | Tolerate clock skew | Stricter sync |
| `gap_warn_s` | 1.0 | s | Warn on pose gaps larger than this | Fewer gap warnings | Flag brief dropouts |
| `alignment_max_dt_s` | 0.02 | s | Association window for ATE/RPE | More matches on sparse GT | Stricter pairing |

### `default_covariance`

Used when a source omits covariance (common for CERBERUS TUM exports).

| Field | Default | Unit | What it does | Raise | Lower |
| --- | --- | --- | --- | --- | --- |
| `std_xy_m` | 0.05 | m | Claimed horizontal σ | Larger naive PL, lower MDR | Tighter PL, higher MDR |
| `std_z_m` | 0.05 | m | Claimed vertical σ | Same tradeoff on VPL | Same |
| `std_rot_rad` | 0.05 | rad | Claimed orientation σ per axis | Larger rotational uncertainty | Rarely drives HPL |

### `per_source_covariance`

Map of source name → same fields as `default_covariance`. Empty by default.
Override per estimator when you know one source is looser than another.

### `monitor`

| Field | Default | Unit | What it does |
| --- | --- | --- | --- |
| `mode` | `consistency` | enum | `naive` \| `consistency` \| `degeneracy`. Product path is `consistency`. |
| `primary_source` | `null` | str or null | For naive/degeneracy, which dict key drives PL (`null` = auto). |

### `evaluation`

| Field | Default | Meaning |
| --- | --- | --- |
| `align_mode` | `umeyama` | `none` \| `origin` \| `umeyama` before true-error scoring |

### `diagnostics`

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | false | Write JSONL diagnostics when true |
| `path` | null | Output path; required if enabled |


## Worked example: conservative safety-oriented profile

Illustrative only. Goal: prefer larger bounds and earlier distrust over
availability.

```yaml
integrity:
  chi2_confidence: 0.999
  p_fa: 0.10          # more willing to alarm
  pl_confidence: 0.999
  alert_limit_m: 1.0  # stricter ops AL
  untrusted_pl_m: 1000000.0

default_covariance:
  std_xy_m: 0.10      # more humble claimed σ if cov missing
  std_z_m: 0.10
  std_rot_rad: 0.05

monitor:
  mode: consistency

window:
  holdover_max_dt_s: 0.2  # less willing to invent missing sources
```

Expect: higher FAR, lower availability, hopefully lower MDR on independent-source
faults. Does **not** fix common-mode corruption.

## Worked example: permissive / availability-oriented profile

Illustrative only. Goal: stay nominal longer; accept more MDR risk.

```yaml
integrity:
  chi2_confidence: 0.99
  p_fa: 0.01
  pl_confidence: 0.99
  alert_limit_m: 3.0

default_covariance:
  std_xy_m: 0.03
  std_z_m: 0.03
  std_rot_rad: 0.05

monitor:
  mode: consistency

window:
  holdover_max_dt_s: 1.0
```

Expect: tighter PLs, fewer alarms, higher availability, higher MDR risk if
estimators are overconfident. On clean CompSLAM, sigma-sweep work already showed
a well-chosen static σ can look fine; permissiveness is not free on faults.

## Loading in code

```python
from cairn import CairnConfig, CairnMonitor

cfg = CairnConfig.from_yaml("configs/local.yaml")
with CairnMonitor(cfg) as mon:
    ...
```
