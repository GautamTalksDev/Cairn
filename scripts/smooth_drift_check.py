"""No-drift check for the drift timing result.

Runs the repo's own consistency monitor and alarm duty metric on four versions
of the prepared CompSLAM sequence: the published injected drift, a smooth
along-track drift, a fixed direction drift, and the clean recording with no
drift at all. If the clean run crosses the same 5 s / 50% alarm duty threshold
at about the same time as the drifted run, that timing is not evidence of drift
detection.

Needs prepared data/processed/ (scripts/prepare_cerberus.py, then
scripts/inject_faults.py for the drift variant).

Usage:
    python scripts/smooth_drift_check.py [out.json]

Takes about a minute on a laptop. Default output: outputs/smooth_drift_check.json
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location("dt", REPO / "scripts" / "drift_timing.py")
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)
from cairn._internal.baselines.naive import with_claimed_diagonal_covariance  # noqa: E402
from cairn._internal.io.tum import load_tum  # noqa: E402
from cairn.cli import true_errors_at_reports  # noqa: E402
from cairn.types import IntegrityState, Pose, PoseEstimate, Trajectory  # noqa: E402

clean = load_tum(REPO / "data/processed/anymal_1_1_compslam/est.tum")
gt = load_tum(REPO / "data/processed/anymal_1_1_compslam/gt.tum")
repo_drift = load_tum(REPO / "data/processed/anymal_1_1_compslam_drift/est.tum")
sigma = 0.06938151515151515
rate, start_frac = 0.02, 0.30
n = len(clean)
i0 = int(n * start_frac)
t = np.array([e.pose.timestamp for e in clean])
t0 = t[i0]
P = np.array([e.pose.position for e in clean])


def smooth_headings(base=30, min_disp=0.10):
    h = np.zeros((n, 2))
    last = np.array([1.0, 0.0])
    for i in range(n):
        j = max(0, i - base)
        d = P[i, :2] - P[j, :2]
        nd = np.linalg.norm(d)
        if nd >= min_disp:
            last = d / nd
        h[i] = last
    return h


def make(offsets):
    return Trajectory(
        [
            PoseEstimate(
                pose=Pose(
                    timestamp=pe.pose.timestamp,
                    position=pe.pose.position + offsets[i],
                    orientation=pe.pose.orientation,
                    frame=pe.pose.frame,
                ),
                covariance=pe.covariance.copy(),
                source="x",
            )
            for i, pe in enumerate(clean)
        ]
    )


mag = np.where(np.arange(n) >= i0, rate * np.maximum(t - t0, 0), 0.0)
hs = smooth_headings()
off_smooth = np.zeros((n, 3))
off_smooth[:, :2] = mag[:, None] * hs
off_fixed = np.zeros((n, 3))
off_fixed[:, 0] = mag
variants = {
    "repo_drift(as published)": repo_drift,
    "smooth_along_track(3s heading)": make(off_smooth),
    "fixed_direction": make(off_fixed),
    "clean(no drift)": clean,
}
out = {}
for name, est in variants.items():
    con = dt._consistency_reports(est, sigma=sigma)
    nai = dt._naive_reports(est, sigma=sigma, confidence=0.999)
    est_c = with_claimed_diagonal_covariance(est, std_xy=sigma, std_z=sigma, source="t")
    err = true_errors_at_reports(est_c, gt, nai, align_mode="umeyama", max_dt=0.05)
    res = {}
    for mname, reps in (("consistency", con), ("naive", nai)):
        d = dt._alarm_duty_crossing(reps, err, t_onset_abs=t0, t0_abs=t[0])
        al = np.array([r.state != IntegrityState.NOMINAL for r in reps])
        res[mname] = {
            "detected": d["detected"],
            "dt_s": d.get("time_to_alarm_s"),
            "err_at_detect_m": d.get("true_error_at_alarm_m"),
            "alarm_frac_pre": float(al[:i0].mean()),
            "alarm_frac_post": float(al[i0:].mean()),
        }
    res["max_err_m"] = float(np.max(err))
    out[name] = res
    print(name, json.dumps(res), flush=True)
out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "smooth_drift_check.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print("wrote", out_path)
