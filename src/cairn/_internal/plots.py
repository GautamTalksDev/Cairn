"""Matplotlib plotting helpers for Cairn integrity reports."""

from __future__ import annotations

__all__ = [
    "plot_error_vs_pl",
    "plot_stanford",
    "plot_trajectories",
]


from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from cairn.metrics.integrity import IntegrityReportSummary, StanfordDiagramData
from cairn.types import IntegrityReport, IntegrityState, Trajectory

_STATE_FACE = {
    IntegrityState.NOMINAL: "#c8e6c9",
    IntegrityState.DEGRADED: "#ffe0b2",
    IntegrityState.DO_NOT_TRUST: "#ffcdd2",
}


def plot_stanford(
    summary: IntegrityReportSummary | StanfordDiagramData,
    ax: Axes | None = None,
) -> Axes:
    """Draw a Stanford-ESA integrity diagram with quadrant boundaries and labels."""
    data = summary.stanford if isinstance(summary, IntegrityReportSummary) else summary
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)

    al = float(data.alert_limit)
    x = np.asarray(data.true_errors, dtype=np.float64)
    y = np.asarray(data.protection_levels, dtype=np.float64)

    # Axis limits: leave room past AL for labels.
    max_err = float(x.max()) if x.size else al
    max_pl = float(y.max()) if y.size else al
    lim = max(al * 1.35, max_err * 1.15, max_pl * 1.15, 1.0)

    # Quadrant background panels (mutually exclusive regions).
    ax.axhspan(al, lim, xmin=0.0, xmax=1.0, color="#eceff1", zorder=0)  # unavailable
    ax.fill_between([al, lim], 0.0, al, color="#ffebee", zorder=0, alpha=0.95)  # HMI
    # Nominal: above diagonal inside the AL box; MI: below diagonal.
    diag = np.linspace(0.0, al, 64)
    ax.fill_between(diag, diag, al, color="#e8f5e9", zorder=0, alpha=0.95)
    ax.fill_between(diag, 0.0, diag, color="#fff3e0", zorder=0, alpha=0.95)

    # Boundaries.
    ax.plot([0.0, lim], [0.0, lim], color="#455a64", lw=1.6, ls="--", label="PE = PL")
    ax.axvline(al, color="#1565c0", lw=2.0, label=f"Alert limit ({al:g} m)")
    ax.axhline(al, color="#1565c0", lw=2.0)

    if x.size:
        colors = []
        for err, pl in zip(x, y, strict=True):
            if pl >= al:
                colors.append("#607d8b")  # unavailable
            elif err > al:
                colors.append("#c62828")  # HMI
            elif err > pl:
                colors.append("#ef6c00")  # MI
            else:
                colors.append("#2e7d32")  # nominal
        ax.scatter(x, y, c=colors, s=42, edgecolors="white", linewidths=0.6, zorder=3)

    # Slide-legible region labels.
    ax.text(
        0.28 * al,
        0.72 * al,
        "Nominal\n(PE ≤ PL < AL)",
        ha="center",
        va="center",
        fontsize=11,
        color="#1b5e20",
        fontweight="bold",
    )
    ax.text(
        0.72 * al,
        0.28 * al,
        "Misleading\n(PE > PL)",
        ha="center",
        va="center",
        fontsize=11,
        color="#e65100",
        fontweight="bold",
    )
    ax.text(
        min(0.5 * (al + lim), lim * 0.78),
        0.45 * al,
        "HMI\n(PE > AL, PL < AL)",
        ha="center",
        va="center",
        fontsize=11,
        color="#b71c1c",
        fontweight="bold",
    )
    ax.text(
        0.5 * lim,
        min(0.5 * (al + lim), lim * 0.88),
        "Unavailable\n(PL ≥ AL)",
        ha="center",
        va="center",
        fontsize=11,
        color="#37474f",
        fontweight="bold",
    )

    counts = (
        f"N={data.nominal_count}  MI={data.misleading_count}  "
        f"HMI={data.hazardously_misleading_count}  UA={data.unavailable_count}"
    )
    ax.set_title("Stanford integrity diagram", fontsize=14, pad=10)
    ax.text(
        0.5,
        1.02,
        counts,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#37474f",
    )
    ax.set_xlabel("True position error PE (m)", fontsize=12)
    ax.set_ylabel("Protection level PL (m)", fontsize=12)
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", color="#cfd8dc", lw=0.8)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    if created_fig:
        ax.figure.set_facecolor("white")
    return ax


def plot_error_vs_pl(
    reports: Sequence[IntegrityReport],
    true_errors: Sequence[float] | NDArray[np.floating],
    ax: Axes | None = None,
) -> Axes:
    """Time series of true error and protection level, shaded by integrity state."""
    if len(reports) != len(true_errors):
        raise ValueError(
            f"reports/true_errors length mismatch: {len(reports)} vs {len(true_errors)}"
        )
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)

    times = np.array([r.timestamp for r in reports], dtype=np.float64)
    pls = np.array([r.protection_level_horizontal for r in reports], dtype=np.float64)
    errors = np.asarray(true_errors, dtype=np.float64)

    # Shade contiguous state runs.
    if len(reports) > 0:
        start = 0
        for i in range(1, len(reports) + 1):
            if i == len(reports) or reports[i].state is not reports[start].state:
                t0 = times[start]
                t1 = times[i - 1] if i - 1 > start else times[start]
                # Expand a tiny bit so single-sample states remain visible.
                if t1 <= t0:
                    pad = 0.01 if len(times) == 1 else 0.5 * np.median(np.diff(times))
                    t0 = float(t0 - pad)
                    t1 = float(t1 + pad)
                state = reports[start].state
                ax.axvspan(
                    t0,
                    t1,
                    color=_STATE_FACE[state],
                    alpha=0.85,
                    lw=0,
                    zorder=0,
                )
                start = i

    ax.plot(times, errors, color="#0d47a1", lw=2.0, label="True error")
    ax.plot(times, pls, color="#6a1b9a", lw=2.0, ls="--", label="Protection level")

    # State legend proxies.
    for state, label in (
        (IntegrityState.NOMINAL, "NOMINAL"),
        (IntegrityState.DEGRADED, "DEGRADED"),
        (IntegrityState.DO_NOT_TRUST, "DO_NOT_TRUST"),
    ):
        ax.axvspan(
            np.nan,
            np.nan,
            color=_STATE_FACE[state],
            label=label,
        )

    ax.set_title("True error vs protection level", fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Metres", fontsize=11)
    ax.grid(True, color="#cfd8dc", lw=0.8)
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.95)
    if created_fig:
        ax.figure.set_facecolor("white")
    return ax


def plot_trajectories(
    est: Trajectory,
    gt: Trajectory,
    axs: tuple[Axes, Axes] | None = None,
) -> tuple[Axes, Axes]:
    """Top-down XY path plus Z-versus-time."""
    if axs is None:
        fig, created = plt.subplots(
            1,
            2,
            figsize=(10.5, 4.5),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.15, 1.0]},
        )
        ax_xy, ax_z = created
        fig.set_facecolor("white")
    else:
        ax_xy, ax_z = axs

    est_xy = np.array([e.pose.position[:2] for e in est], dtype=np.float64)
    gt_xy = np.array([e.pose.position[:2] for e in gt], dtype=np.float64)
    est_z = np.array([e.pose.position[2] for e in est], dtype=np.float64)
    gt_z = np.array([e.pose.position[2] for e in gt], dtype=np.float64)
    est_t = est.timestamps()
    gt_t = gt.timestamps()

    ax_xy.plot(gt_xy[:, 0], gt_xy[:, 1], color="#455a64", lw=2.2, label="Ground truth")
    ax_xy.plot(est_xy[:, 0], est_xy[:, 1], color="#1565c0", lw=1.8, label="Estimate")
    ax_xy.scatter(gt_xy[0, 0], gt_xy[0, 1], c="#455a64", s=36, zorder=3)
    ax_xy.scatter(est_xy[0, 0], est_xy[0, 1], c="#1565c0", s=36, zorder=3)
    ax_xy.set_title("Top-down trajectory (XY)", fontsize=13)
    ax_xy.set_xlabel("X (m)", fontsize=11)
    ax_xy.set_ylabel("Y (m)", fontsize=11)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, color="#cfd8dc", lw=0.8)
    ax_xy.legend(loc="best", fontsize=9)

    ax_z.plot(gt_t, gt_z, color="#455a64", lw=2.2, label="Ground truth")
    ax_z.plot(est_t, est_z, color="#1565c0", lw=1.8, label="Estimate")
    ax_z.set_title("Vertical profile (Z vs time)", fontsize=13)
    ax_z.set_xlabel("Time (s)", fontsize=11)
    ax_z.set_ylabel("Z (m)", fontsize=11)
    ax_z.grid(True, color="#cfd8dc", lw=0.8)
    ax_z.legend(loc="best", fontsize=9)
    return ax_xy, ax_z
