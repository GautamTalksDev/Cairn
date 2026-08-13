"""CP-17 — Streaming monitor p99 latency gate (body-worn budget).

Fails if any monitor's p99 ``update()`` latency exceeds 100 ms on the
3-source case. Marked ``slow`` (excluded from default CI via ``-m "not slow"``).

Run::

    pytest -m slow tests/test_benchmark.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET_MS = 100.0


def _load_benchmark():
    import sys

    path = REPO_ROOT / "scripts" / "benchmark.py"
    name = "cairn_benchmark"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Required so @dataclass in the script can resolve annotations.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_p99_latency_under_100ms_three_sources() -> None:
    """Body-worn gate: 3-source p99 must stay under 100 ms for every monitor."""
    bench = _load_benchmark()
    # Fewer epochs than the published report, still enough for a stable p99.
    results = bench.run_benchmark(
        epochs=800,
        warmup=50,
        budget_ms=BUDGET_MS,
        source_counts=(3,),
        monitors=("naive", "consistency", "degeneracy"),
        seed=0,
        profile_on_over_budget=False,
    )
    assert len(results) == 3
    failures: list[str] = []
    for r in results:
        assert r.n_sources == 3
        if r.latency.p99_ms > BUDGET_MS:
            failures.append(
                f"{r.mode}: p99={r.latency.p99_ms:.3f} ms > {BUDGET_MS:.0f} ms "
                f"(p50={r.latency.p50_ms:.3f}, p95={r.latency.p95_ms:.3f})"
            )
    assert not failures, "p99 budget exceeded:\n" + "\n".join(failures)
