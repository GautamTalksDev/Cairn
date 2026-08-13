#!/usr/bin/env python3
"""CP-19 — Reproduce every published experiment from one command.

Runs end-to-end (in order)::

    sigma sweep → fault injection → ROC → multi-sequence → drift timing →
    timing robustness → common-mode → benchmark → RESULTS.md

Writes figures / tables under ``outputs/`` and pins the exact code + environment
in ``outputs/provenance.json``.

Example::

    python scripts/reproduce_all.py
    rm -rf outputs && python scripts/reproduce_all.py

Fails loudly if required input data is missing (does **not** silently fall
back to the quickstart subsample).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np

# Master seed for every RNG-using experiment (also exported as CAIRN_RNG_SEED).
DEFAULT_SEED = 0

CLEAN_SEQ = "anymal_1_1_compslam"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    argv: list[str]
    requires: tuple[Path, ...]
    produces: tuple[Path, ...]
    seed: int


def _git_info(root: Path) -> dict[str, object]:
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(
                args, cwd=root, stderr=subprocess.DEVNULL, text=True
            )
            return out.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    describe = _run(["git", "describe", "--always", "--dirty", "--tags"])
    return {
        "commit": commit,
        "describe": describe,
        "dirty": bool(dirty) if dirty is not None else None,
        "status_porcelain": dirty if dirty else "",
    }


def _source_tree_hash(root: Path) -> str:
    """Content hash of tracked library/scripts when git is unavailable or dirty."""
    h = hashlib.sha256()
    roots = [root / "src" / "cairn", root / "scripts", root / "configs"]
    files: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".toml", ".typed"}:
                files.append(path)
    for path in files:
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _dependency_versions() -> dict[str, str]:
    names = sorted(
        {
            "cairn",
            "numpy",
            "scipy",
            "matplotlib",
            "PyYAML",
            "pytest",
            "mypy",
            "ruff",
        }
    )
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    # Full freeze for exact pin.
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        out["_pip_freeze"] = freeze.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        out["_pip_freeze"] = ""
    return out


def _require(paths: list[Path], *, label: str) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(
            f"ERROR: missing required {label}:\n  - "
            + "\n  - ".join(missing)
            + "\nPrepare data first (e.g. scripts/prepare_cerberus.py), "
            "then re-run. Refusing to skip."
        )


def _build_steps(root: Path, *, seed: int) -> list[Step]:
    py = sys.executable
    scripts = root / "scripts"
    processed = root / "data" / "processed"
    clean = processed / CLEAN_SEQ
    drift = processed / f"{CLEAN_SEQ}_drift"
    outputs = root / "outputs"

    return [
        Step(
            name="sigma_sweep",
            argv=[
                py,
                str(scripts / "sigma_sweep.py"),
                "--sequence-dir",
                str(clean),
                "--out-dir",
                str(outputs / "sigma_sweep"),
            ],
            requires=(clean / "est.tum", clean / "gt.tum"),
            produces=(
                outputs / "sigma_sweep" / "sigma_sweep.json",
                outputs / "sigma_sweep" / "mdr_far_vs_sigma.png",
                outputs / "sigma_sweep" / "summary.md",
            ),
            seed=seed,
        ),
        Step(
            name="inject_faults",
            argv=[
                py,
                str(scripts / "inject_faults.py"),
                "--source",
                str(clean),
                "--faults",
                "all",
                "--out-root",
                str(processed),
            ],
            requires=(clean / "est.tum", clean / "gt.tum"),
            produces=(
                processed / f"{CLEAN_SEQ}_bias" / "est.tum",
                processed / f"{CLEAN_SEQ}_dropout" / "est.tum",
                drift / "est.tum",
            ),
            seed=seed,
        ),
        Step(
            name="roc_curve",
            argv=[
                py,
                str(scripts / "roc_curve.py"),
                "--sequence-dir",
                str(clean),
                "--out-dir",
                str(outputs / "roc"),
            ],
            requires=(
                clean / "est.tum",
                clean / "gt.tum",
                outputs / "sigma_sweep" / "sigma_sweep.json",
            ),
            produces=(
                outputs / "roc" / "roc.json",
                outputs / "roc" / "roc.png",
                outputs / "roc" / "summary.md",
            ),
            seed=seed,
        ),
        Step(
            name="multiseq",
            argv=[
                py,
                str(scripts / "run_all_sequences.py"),
                "--processed-root",
                str(processed),
                "--out-dir",
                str(outputs / "multiseq"),
            ],
            requires=(
                clean / "est.tum",
                drift / "est.tum",
                outputs / "sigma_sweep" / "sigma_sweep.json",
            ),
            produces=(
                outputs / "multiseq" / "multiseq.json",
                outputs / "multiseq" / "summary.md",
            ),
            seed=seed,
        ),
        Step(
            name="drift_timing",
            argv=[
                py,
                str(scripts / "drift_timing.py"),
                "--parent",
                str(clean),
                "--drift",
                str(drift),
                "--out-dir",
                str(outputs / "drift_timing"),
            ],
            requires=(
                clean / "est.tum",
                drift / "est.tum",
                drift / "gt.tum",
                outputs / "sigma_sweep" / "sigma_sweep.json",
            ),
            produces=(
                outputs / "drift_timing" / "timing.json",
                outputs / "drift_timing" / "drift_timing.png",
                outputs / "drift_timing" / "summary.md",
            ),
            seed=seed,
        ),
        Step(
            name="timing_robustness",
            argv=[
                py,
                str(scripts / "timing_robustness.py"),
                "--parent",
                str(clean),
                "--drift",
                str(drift),
                "--out-dir",
                str(outputs / "timing_robustness"),
            ],
            requires=(
                clean / "est.tum",
                drift / "est.tum",
                outputs / "sigma_sweep" / "sigma_sweep.json",
            ),
            produces=(
                outputs / "timing_robustness" / "robustness.json",
                outputs / "timing_robustness" / "robustness.md",
            ),
            seed=seed,
        ),
        Step(
            name="common_mode",
            argv=[
                py,
                str(scripts / "common_mode_fault.py"),
                "--source",
                str(clean),
                "--out-dir",
                str(outputs / "common_mode"),
            ],
            requires=(
                clean / "est.tum",
                clean / "gt.tum",
                outputs / "sigma_sweep" / "sigma_sweep.json",
            ),
            produces=(
                outputs / "common_mode" / "common_mode.json",
                outputs / "common_mode" / "common_mode.png",
                outputs / "common_mode" / "summary.md",
            ),
            seed=seed,
        ),
        Step(
            name="benchmark",
            argv=[
                py,
                str(scripts / "benchmark.py"),
                "--out-dir",
                str(outputs / "benchmark"),
                "--seed",
                str(seed),
            ],
            requires=(),
            produces=(
                outputs / "benchmark" / "benchmark.md",
                outputs / "benchmark" / "benchmark.json",
            ),
            seed=seed,
        ),
        Step(
            name="results_md",
            argv=[
                py,
                str(scripts / "generate_results.py"),
                "--sequence-dir",
                str(clean),
                "--out",
                str(root / "RESULTS.md"),
            ],
            requires=(clean / "est.tum", clean / "gt.tum"),
            produces=(root / "RESULTS.md",),
            seed=seed,
        ),
    ]


def _run_step(step: Step, *, root: Path, env: dict[str, str]) -> dict[str, object]:
    print(f"\n=== [{step.name}] ===", flush=True)
    print(" ".join(step.argv), flush=True)
    if step.requires:
        _require(list(step.requires), label=f"inputs for {step.name}")

    t0 = time.perf_counter()
    proc = subprocess.run(
        step.argv,
        cwd=root,
        env=env,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(
            f"ERROR: step {step.name!r} failed with exit code {proc.returncode}"
        )

    missing = [str(p) for p in step.produces if not p.is_file()]
    if missing:
        raise SystemExit(
            f"ERROR: step {step.name!r} finished but missing outputs:\n  - "
            + "\n  - ".join(missing)
        )

    print(f"[{step.name}] ok in {elapsed:.1f}s", flush=True)
    return {
        "name": step.name,
        "argv": step.argv,
        "seed": step.seed,
        "elapsed_s": elapsed,
        "produces": [str(p.relative_to(root)) for p in step.produces],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Master RNG seed (default: {DEFAULT_SEED}); exported as CAIRN_RNG_SEED",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the latency benchmark step",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only these step names (debug); still fails if their inputs are missing",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    clean = root / "data" / "processed" / CLEAN_SEQ
    _require(
        [clean / "est.tum", clean / "gt.tum"],
        label=f"clean sequence under {clean}",
    )

    seed = int(args.seed)
    os.environ["CAIRN_RNG_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = os.environ.copy()
    env["CAIRN_RNG_SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)
    # Prefer the project venv interpreter's env; ensure PYTHONPATH includes src.
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    steps = _build_steps(root, seed=seed)
    if args.skip_benchmark:
        steps = [s for s in steps if s.name != "benchmark"]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {s.name for s in steps}
        if unknown:
            raise SystemExit(f"ERROR: unknown step(s): {sorted(unknown)}")
        steps = [s for s in steps if s.name in wanted]

    started = datetime.now(UTC)
    wall0 = time.perf_counter()
    step_records: list[dict[str, object]] = []
    for step in steps:
        step_records.append(_run_step(step, root=root, env=env))
    wall = time.perf_counter() - wall0
    finished = datetime.now(UTC)

    git = _git_info(root)
    provenance = {
        "created_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "total_runtime_s": wall,
        "seed": seed,
        "cairn_rng_seed_env": env["CAIRN_RNG_SEED"],
        "pythonhashseed_env": env["PYTHONHASHSEED"],
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": git,
        "source_tree_sha256": _source_tree_hash(root),
        "dependencies": _dependency_versions(),
        "numpy": np.__version__,
        "clean_sequence": str(clean.relative_to(root)),
        "steps": step_records,
        "command": [sys.executable, "scripts/reproduce_all.py", *sys.argv[1:]],
    }

    prov_path = outputs / "provenance.json"
    # Drop huge freeze into a sibling for readability of the main pin file.
    freeze = provenance["dependencies"].pop("_pip_freeze", "")  # type: ignore[union-attr]
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (outputs / "pip_freeze.txt").write_text(str(freeze) + "\n", encoding="utf-8")

    # Per-step seed stamp (idempotent overwrite).
    for step in steps:
        for out in step.produces:
            parent = out.parent
            if parent.is_dir() and parent != root:
                stamp = {
                    "step": step.name,
                    "seed": step.seed,
                    "cairn_rng_seed": seed,
                    "git_commit": git.get("commit"),
                    "reproduced_utc": finished.isoformat(),
                }
                (parent / "seeds.json").write_text(
                    json.dumps(stamp, indent=2) + "\n", encoding="utf-8"
                )

    print("\n========== REPRODUCE COMPLETE ==========")
    print(f"total runtime: {wall:.1f}s ({wall / 60.0:.2f} min)")
    print(f"seed: {seed}")
    print(f"git: {git.get('commit')}")
    print(f"wrote {prov_path}")
    print(f"wrote {outputs / 'pip_freeze.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
