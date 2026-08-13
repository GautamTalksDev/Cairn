#!/usr/bin/env python3
"""CP-17 — Streaming monitor latency / RSS benchmark (body-worn compute budget).

Measures ``CairnMonitor.update()`` only (``time.perf_counter``), excluding I/O.
Reports p50 / p95 / p99 latency (ms) and peak RSS, by monitor mode and by
source count (2, 3, 5, 8).

Example::

    python scripts/benchmark.py
    python scripts/benchmark.py --out-dir outputs/benchmark --epochs 2000

If any configuration exceeds the p99 budget (default 100 ms), cProfile hotspots
are appended to the report. No optimisation is performed here — measure only.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from cairn.config import CairnConfig, DiagnosticsConfig, MonitorSection
from cairn.monitor import CairnMonitor
from cairn.types import Pose, PoseEstimate

MonitorMode = Literal["naive", "consistency", "degeneracy"]

SOURCE_COUNTS = (2, 3, 5, 8)
MONITORS: tuple[MonitorMode, ...] = ("naive", "consistency", "degeneracy")
DEFAULT_BUDGET_MS = 100.0
DEFAULT_EPOCHS = 2000
DEFAULT_WARMUP = 100
DT_S = 0.05


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def machine_spec() -> dict[str, str]:
    """Collect reproducible machine / runtime identifiers."""
    uname = platform.uname()
    cpu = uname.processor or platform.processor() or "unknown"
    # Prefer Linux model name when available.
    model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    mem_kb = ""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                mem_kb = line.split()[1]
                break
    return {
        "python": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "processor": cpu,
        "cpu_model": model or cpu,
        "cpu_count_logical": str(os.cpu_count() or ""),
        "mem_total_kib": mem_kb,
        "numpy": np.__version__,
    }


def peak_rss_bytes() -> int:
    """Peak resident set size for this process (bytes)."""
    # Linux: ru_maxrss is KiB; macOS: bytes.
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def _pose_estimate(
    t: float,
    *,
    source: str,
    xyz: np.ndarray,
    std: float = 0.05,
) -> PoseEstimate:
    cov = np.diag([std**2, std**2, std**2, 0.01, 0.01, 0.01])
    return PoseEstimate(
        pose=Pose(
            timestamp=float(t),
            position=np.asarray(xyz, dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
        covariance=cov,
        source=source,
    )


def build_epochs(
    n_sources: int,
    n_epochs: int,
    *,
    seed: int = 0,
    dt: float = DT_S,
) -> list[tuple[float, dict[str, PoseEstimate], np.ndarray | None]]:
    """Pre-build in-memory epochs so timing excludes synthetic I/O / allocation."""
    rng = np.random.default_rng(seed + 17 * n_sources)
    names = [f"src_{i}" for i in range(n_sources)]
    epochs: list[tuple[float, dict[str, PoseEstimate], np.ndarray | None]] = []
    for k in range(n_epochs):
        t = k * dt
        ests: dict[str, PoseEstimate] = {}
        for i, name in enumerate(names):
            # Small independent noise so consistency stays mostly NOMINAL.
            xyz = np.array(
                [0.01 * k + 0.002 * i, 0.001 * rng.normal(), 0.0],
                dtype=np.float64,
            )
            ests[name] = _pose_estimate(t, source=name, xyz=xyz)
        # Degeneracy scores: mostly well-constrained, occasional weak axis.
        scores = np.clip(0.55 + 0.35 * rng.random(6), 0.05, 1.0)
        if k % 37 == 0:
            scores[0] = 0.12
        epochs.append((t, ests, scores.astype(np.float64)))
    return epochs


@dataclass(frozen=True, slots=True)
class LatencyStats:
    n_timed: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float

    @classmethod
    def from_seconds(cls, samples_s: np.ndarray) -> LatencyStats:
        ms = np.asarray(samples_s, dtype=np.float64) * 1e3
        if ms.size == 0:
            raise ValueError("no timed samples")
        return cls(
            n_timed=int(ms.size),
            p50_ms=float(np.percentile(ms, 50)),
            p95_ms=float(np.percentile(ms, 95)),
            p99_ms=float(np.percentile(ms, 99)),
            mean_ms=float(np.mean(ms)),
            max_ms=float(np.max(ms)),
        )


@dataclass(slots=True)
class BenchResult:
    mode: MonitorMode
    n_sources: int
    latency: LatencyStats
    peak_rss_mib: float
    rss_delta_mib: float
    over_budget: bool
    profile_top: list[dict[str, float | str]] = field(default_factory=list)


def _make_monitor(mode: MonitorMode, n_sources: int) -> CairnMonitor:
    from dataclasses import replace

    cfg = replace(
        CairnConfig(),
        monitor=MonitorSection(
            mode=mode,
            primary_source="src_0" if mode in ("naive", "degeneracy") else None,
        ),
        diagnostics=DiagnosticsConfig(enabled=False, path=None),
        window=replace(
            CairnConfig().window,
            max_epochs=max(50, n_sources * 10),
            max_seconds=10.0,
            holdover_max_dt_s=0.0,
        ),
    )
    return CairnMonitor(cfg)


def _run_timed(
    mode: MonitorMode,
    epochs: list[tuple[float, dict[str, PoseEstimate], np.ndarray | None]],
    *,
    warmup: int,
) -> tuple[np.ndarray, float, float]:
    """Return (latencies_s after warmup, rss_before_mib, peak_rss_mib)."""
    mon = _make_monitor(mode, n_sources=len(epochs[0][1]))
    # Touch once so imports / first-call costs land in warmup.
    t0, e0, s0 = epochs[0]
    if mode == "degeneracy":
        mon.update(t0, e0, degeneracy_scores=s0)
    else:
        mon.update(t0, e0)
    mon.reset()

    rss_before = peak_rss_bytes() / (1024 * 1024)
    samples: list[float] = []
    for i, (ts, ests, scores) in enumerate(epochs):
        if mode == "degeneracy":
            t_a = time.perf_counter()
            mon.update(ts, ests, degeneracy_scores=scores)
            t_b = time.perf_counter()
        else:
            t_a = time.perf_counter()
            mon.update(ts, ests)
            t_b = time.perf_counter()
        if i >= warmup:
            samples.append(t_b - t_a)
    mon.close()
    rss_peak = peak_rss_bytes() / (1024 * 1024)
    return np.asarray(samples, dtype=np.float64), rss_before, rss_peak


def profile_update(
    mode: MonitorMode,
    epochs: list[tuple[float, dict[str, PoseEstimate], np.ndarray | None]],
    *,
    n_profile: int = 400,
    top_n: int = 15,
) -> list[dict[str, float | str]]:
    """cProfile ``update()`` over ``n_profile`` epochs; return top hotspots."""
    mon = _make_monitor(mode, n_sources=len(epochs[0][1]))
    subset = epochs[:n_profile]
    pr = cProfile.Profile()
    pr.enable()
    for ts, ests, scores in subset:
        if mode == "degeneracy":
            mon.update(ts, ests, degeneracy_scores=scores)
        else:
            mon.update(ts, ests)
    pr.disable()
    mon.close()

    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    # Collect top entries programmatically.
    rows: list[dict[str, float | str]] = []
    for (filename, line, func), (_cc, nc, tt, ct, _callers) in sorted(
        stats.stats.items(),
        key=lambda kv: kv[1][3],
        reverse=True,
    )[:top_n]:
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}:{func}",
                "ncalls": float(nc),
                "tottime_s": float(tt),
                "cumtime_s": float(ct),
            }
        )
    return rows


def run_benchmark(
    *,
    epochs: int = DEFAULT_EPOCHS,
    warmup: int = DEFAULT_WARMUP,
    budget_ms: float = DEFAULT_BUDGET_MS,
    source_counts: tuple[int, ...] = SOURCE_COUNTS,
    monitors: tuple[MonitorMode, ...] = MONITORS,
    seed: int = 0,
    profile_on_over_budget: bool = True,
) -> list[BenchResult]:
    """Run the full matrix and return structured results."""
    if warmup < 0 or epochs <= warmup:
        raise ValueError(f"need epochs > warmup >= 0 (got epochs={epochs}, warmup={warmup})")

    results: list[BenchResult] = []
    for n_src in source_counts:
        built = build_epochs(n_src, epochs, seed=seed)
        for mode in monitors:
            samples, rss_before, rss_peak = _run_timed(mode, built, warmup=warmup)
            lat = LatencyStats.from_seconds(samples)
            over = lat.p99_ms > budget_ms
            profile_top: list[dict[str, float | str]] = []
            if over and profile_on_over_budget:
                profile_top = profile_update(mode, built)
            results.append(
                BenchResult(
                    mode=mode,
                    n_sources=n_src,
                    latency=lat,
                    peak_rss_mib=rss_peak,
                    rss_delta_mib=rss_peak - rss_before,
                    over_budget=over,
                    profile_top=profile_top,
                )
            )
    return results


def _fmt_ms(x: float) -> str:
    return f"{x:.3f}"


def render_markdown(
    results: list[BenchResult],
    *,
    spec: dict[str, str],
    epochs: int,
    warmup: int,
    budget_ms: float,
    seed: int,
) -> str:
    lines: list[str] = [
        "# Cairn streaming monitor benchmark (CP-17)",
        "",
        "Per-epoch latency of `CairnMonitor.update()` only "
        "(`time.perf_counter`; synthetic epochs pre-built — no I/O in the timed path).",
        "",
        "## Machine / runtime",
        "",
        f"- Python: `{spec['python']}`",
        f"- Implementation: `{spec['python_implementation']}`",
        f"- Platform: `{spec['platform']}`",
        f"- CPU: `{spec['cpu_model']}` ({spec['cpu_count_logical']} logical)",
        f"- Machine: `{spec['machine']}` / `{spec['system']} {spec['release']}`",
    ]
    if spec.get("mem_total_kib"):
        mem_gib = float(spec["mem_total_kib"]) / (1024 * 1024)
        lines.append(f"- Host RAM: `{mem_gib:.1f} GiB`")
    lines.extend(
        [
            f"- NumPy: `{spec['numpy']}`",
            "",
            "## Method",
            "",
            f"- Timed epochs after warmup: **{epochs - warmup}** "
            f"(total {epochs}, warmup {warmup})",
            f"- Sample period: `{DT_S} s` synthetic",
            f"- RNG seed: `{seed}`",
            f"- p99 budget: **{budget_ms:.0f} ms** "
            "(body-worn gate; failures trigger cProfile)",
            "- Monitors: `naive`, `consistency`, `degeneracy`",
            "- Source counts: `2`, `3`, `5`, `8`",
            "- Peak RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss`",
            "",
            "## Results",
            "",
            "| Monitor | Sources | n | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | max (ms) | "
            "Peak RSS (MiB) | Over budget? |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for r in results:
        flag = "**YES**" if r.over_budget else "no"
        lines.append(
            f"| `{r.mode}` | {r.n_sources} | {r.latency.n_timed} | "
            f"{_fmt_ms(r.latency.p50_ms)} | {_fmt_ms(r.latency.p95_ms)} | "
            f"{_fmt_ms(r.latency.p99_ms)} | {_fmt_ms(r.latency.mean_ms)} | "
            f"{_fmt_ms(r.latency.max_ms)} | {r.peak_rss_mib:.1f} | {flag} |"
        )

    # Headline callouts
    three = [r for r in results if r.n_sources == 3]
    lines.extend(["", "## Headline (3 sources)", ""])
    for r in three:
        lines.append(
            f"- `{r.mode}` p99 = **{_fmt_ms(r.latency.p99_ms)} ms** "
            f"(p50={_fmt_ms(r.latency.p50_ms)}, p95={_fmt_ms(r.latency.p95_ms)})"
        )

    over = [r for r in results if r.over_budget]
    lines.extend(["", "## Profiles (over-budget only)", ""])
    if not over:
        lines.append(
            f"No configuration exceeded the {budget_ms:.0f} ms p99 budget; "
            "cProfile was not required."
        )
    else:
        lines.append(
            "Hotspots from `cProfile` around `update()` "
            "(sorted by cumulative time). **Not optimised in CP-17 — record only.**"
        )
        for r in over:
            lines.extend(
                [
                    "",
                    f"### `{r.mode}` × {r.n_sources} sources "
                    f"(p99={_fmt_ms(r.latency.p99_ms)} ms)",
                    "",
                    "| Function | ncalls | tottime (s) | cumtime (s) |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for row in r.profile_top:
                lines.append(
                    f"| `{row['function']}` | {int(row['ncalls'])} | "
                    f"{float(row['tottime_s']):.6f} | {float(row['cumtime_s']):.6f} |"
                )
            if r.profile_top:
                top = r.profile_top[0]["function"]
                lines.append("")
                lines.append(f"Dominant (by cumtime): **`{top}`**")

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/benchmark.py --out-dir outputs/benchmark",
            "```",
            "",
            "Artifacts: `benchmark.md`, `benchmark.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def results_to_jsonable(
    results: list[BenchResult],
    *,
    spec: dict[str, str],
    epochs: int,
    warmup: int,
    budget_ms: float,
    seed: int,
) -> dict:
    return {
        "machine": spec,
        "method": {
            "epochs": epochs,
            "warmup": warmup,
            "budget_ms": budget_ms,
            "seed": seed,
            "dt_s": DT_S,
            "source_counts": list(SOURCE_COUNTS),
            "monitors": list(MONITORS),
        },
        "results": [
            {
                "mode": r.mode,
                "n_sources": r.n_sources,
                "latency": asdict(r.latency),
                "peak_rss_mib": r.peak_rss_mib,
                "rss_delta_mib": r.rss_delta_mib,
                "over_budget": r.over_budget,
                "profile_top": r.profile_top,
            }
            for r in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "outputs" / "benchmark",
        help="Directory for benchmark.md / benchmark.json",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip cProfile even when over budget",
    )
    args = parser.parse_args(argv)

    seed = (
        int(args.seed)
        if args.seed is not None
        else int(os.environ.get("CAIRN_RNG_SEED", "0"))
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"benchmarking monitors={list(MONITORS)} sources={list(SOURCE_COUNTS)} "
        f"epochs={args.epochs} warmup={args.warmup} …",
        flush=True,
    )
    results = run_benchmark(
        epochs=args.epochs,
        warmup=args.warmup,
        budget_ms=args.budget_ms,
        seed=seed,
        profile_on_over_budget=not args.no_profile,
    )
    spec = machine_spec()
    md = render_markdown(
        results,
        spec=spec,
        epochs=args.epochs,
        warmup=args.warmup,
        budget_ms=args.budget_ms,
        seed=seed,
    )
    (out_dir / "benchmark.md").write_text(md, encoding="utf-8")
    payload = results_to_jsonable(
        results,
        spec=spec,
        epochs=args.epochs,
        warmup=args.warmup,
        budget_ms=args.budget_ms,
        seed=seed,
    )
    (out_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    # Console summary
    for r in results:
        mark = " OVER" if r.over_budget else ""
        print(
            f"  {r.mode:12s} n={r.n_sources}: "
            f"p50={r.latency.p50_ms:7.3f} p95={r.latency.p95_ms:7.3f} "
            f"p99={r.latency.p99_ms:7.3f} ms  "
            f"RSS={r.peak_rss_mib:.1f} MiB{mark}"
        )
    print(f"wrote {out_dir / 'benchmark.md'}")
    print(f"wrote {out_dir / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
