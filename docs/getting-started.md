# Getting started

About fifteen minutes from a fresh clone to feeding Cairn two of your own pose
sources. Commands below were run on Python 3.12.

## 1. Install

```bash
git clone https://github.com/GautamTalksDev/Cairn.git
cd Cairn
python3.12 -m venv .venv
source .venv/bin/activate
pip install .
```

Check the console script:

```bash
cairn --help
```

You should see usage for the `score` subcommand.

## 2. Bundled examples

Quickstart trajectories are **synthetic** (`scripts/make_quickstart_data.py`):
corridor path, dual sources, injected drift. Not CERBERUS.

```bash
python examples/integrate.py
```

Expected shape (timestamps in seconds from t=0; HPL in metres). The script
prints a pre-drift window and a post-drift window so you should see non-nominal
states after onset (~28 s):

```text
integrate.py: 80 epochs from synthetic quickstart (frame=map, units=m/s; ...)
t=25.0s  NOMINAL       HPL=...m  VPL=...m  (...)
...
t=40.0s  DEGRADED      HPL=...m  ...
```

Also:

```bash
python examples/custom_source.py   # wrap a source name Cairn has never seen
python examples/analyse_run.py     # write + summarise diagnostics JSONL
```

`integrate.py` loads synthetic `est.tum` and `gt.tum`, pushes them into
`CairnMonitor`, and prints state plus protection levels. Positions are metres
in frame `map`.

## 3. Score the bundled trajectories

```bash
cairn score \
  --est examples/quickstart/est.tum \
  --gt examples/quickstart/gt.tum \
  --reports examples/quickstart/reports.txt \
  --alert-limit 2.0 \
  --out outputs/quickstart
```

Reports come from the consistency monitor run baked into
`make_quickstart_data.py`, so MDR is typically non-zero (drift segment). Example
console line:

```text
Wrote artifacts to outputs/quickstart (MDR=0.xxxxxx, epochs=700)
```

Open `outputs/quickstart/summary.json` and `outputs/quickstart/report.md`. You
should also get `stanford.png`, `error_vs_pl.png`, and `trajectories.png`.

## 4. Plug in two of your own sources

Copy the pattern in `examples/integrate.py`. Both sources must share one
navigation frame. Units: metres, seconds, w-first quaternion `(w, x, y, z)`,
covariance 6×6 over `[x, y, z, roll, pitch, yaw]` in m² and rad².

```python
from __future__ import annotations

import numpy as np
from cairn import CairnConfig, CairnMonitor, Pose, PoseEstimate

Q = np.array([1.0, 0.0, 0.0, 0.0])  # identity, w-first
COV = np.diag([0.05**2, 0.05**2, 0.05**2, 0.01, 0.01, 0.01])


def estimate(t: float, xyz: list[float], source: str) -> PoseEstimate:
    return PoseEstimate(
        pose=Pose(timestamp=t, position=np.asarray(xyz, float), orientation=Q.copy()),
        covariance=COV.copy(),
        source=source,
    )


mon = CairnMonitor(CairnConfig())  # or CairnConfig.from_yaml("configs/default.yaml")
for k in range(5):
    t = 0.1 * k
    # Replace the xyz values with your estimator outputs in the same frame.
    report = mon.update(
        t,
        {
            "lidar": estimate(t, [0.01 * k, 0.0, 0.0], "lidar"),
            "vio": estimate(t, [0.01 * k + 0.02, 0.0, 0.0], "vio"),
        },
    )
    print(
        f"t={t:.1f}  {report.state.name:12s}  "
        f"HPL={report.protection_level_horizontal:.3f}m  "
        f"reason={report.reason}"
    )
mon.close()
```

If you only have one source, consistency mode cannot cross-check. The monitor
degrades toward `DO_NOT_TRUST` for solution separation when a lone source is
all it sees. Prefer two independent estimators, or switch `monitor.mode` to
`naive` in config if you knowingly want the covariance baseline only.

## 5. Interpreting the output

Each `IntegrityReport` carries:

- `protection_level_horizontal` / `protection_level_vertical` (metres)
- `state`: `NOMINAL`, `DEGRADED`, or `DO_NOT_TRUST`
- `reason`: short machine-oriented explanation
- `time_since_corroboration`: seconds since sources last agreed, when applicable

Reading rule of thumb:

- Stay in `NOMINAL` only if you accept the published PL as your integrity claim.
- `DEGRADED` means use with caution; the bound has grown or consistency is weak.
- `DO_NOT_TRUST` means do not steer on this solution for safety-critical work.

For offline scoring against ground truth, pair reports with true horizontal
error and compute missed detection rate, false alarm rate, and availability
(see [api-reference.md](api-reference.md) and [concepts.md](concepts.md)).

## 6. Optional: your own config

```bash
cp configs/default.yaml configs/local.yaml
# edit thresholds, then:
```

```python
from cairn import CairnConfig, CairnMonitor

mon = CairnMonitor(CairnConfig.from_yaml("configs/local.yaml"))
```

Field meanings and tuning examples: [configuration.md](configuration.md).

## Next

- Ideas behind PL, MDR, and solution separation: [concepts.md](concepts.md)
- Public API details: [api-reference.md](api-reference.md)
- How published numbers were made: [methodology.md](methodology.md)
- What not to claim: [limitations.md](limitations.md)
