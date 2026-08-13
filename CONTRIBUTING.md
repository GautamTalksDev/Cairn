# Contributing to Cairn

Thank you for wanting to help. This file should be enough to set up, change
code, and open a pull request without a side conversation.

## Scope: what we accept

Cairn **assesses integrity**. It consumes pose estimates and reports protection
levels, integrity state, and related metrics. It **does not compute position**.

We welcome, among other things:

- New integrity monitors / detectors (with tests and docs)
- Bug fixes, clearer errors, and safer input handling
- Documentation and reproducibility improvements
- Evaluation scripts and metrics that stay honest about limits
- CI, packaging, and security-hygiene fixes

We will kindly decline, among other things:

- Pull requests that add a SLAM, odometry, or other **position estimator**
- Changes that embed or commit full CERBERUS / other InC-NC datasets
- Public "security fixes" that should have been private disclosures (see
  [SECURITY.md](SECURITY.md))
- Drive-by refactors with no tests or no problem statement

If you are unsure, open a feature-request issue first. Saying no to an estimator
PR is about scope, not about the quality of your robotics work.

## Development setup

Requires Python 3.12+.

```bash
git clone https://github.com/GautamTalksDev/Cairn.git
cd Cairn
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` registers local hooks (ruff with the bandit `S` rules,
detect-secrets, gitleaks, large-file and whitespace checks). CI runs the same
secret and hygiene scanners; skipping hooks with `--no-verify` is discouraged.

```bash
pre-commit run --all-files
```

## Project layout

| Path | What lives there |
| --- | --- |
| `src/cairn/` | Public package: `CairnMonitor`, `CairnConfig`, types, metrics, CLI |
| `src/cairn/_internal/` | Unsupported internals (baselines, detectors, I/O, plots) |
| `configs/` | Default YAML; copy and retune for experiments |
| `examples/` | `integrate.py`, `custom_source.py`, `analyse_run.py`, synthetic quickstart (`make_quickstart_data.py`) |
| `scripts/` | Evaluation, fault injection, reproduce-all, `make_quickstart_data.py` |
| `tests/` | Unit, integration, and `@pytest.mark.slow` goldens |
| `docs/` | Concepts, API, methodology, limitations |
| `data/` | Local raw/processed sequences (gitignored; never commit CERBERUS dumps) |
| `outputs/` | Local figures and tables (gitignored) |

Integrators should import only the public surface (`from cairn import ...`).
Scripts and tests may use `cairn._internal` knowing it can change.

## Tests

### Fast tests (every push)

```bash
pytest -q -m "not slow"
```

Docstrings under `src/cairn/` (excluding `_internal` and `cli`) are collected
as doctests via `--doctest-modules` in `pyproject.toml`. Keep examples
executable; they run on every `pytest` invocation.

### Slow tests (needs `data/processed/`)

```bash
pytest -q -m slow
```

These lock headline numbers against `tests/golden/expected_results.json`.

### Full reproduction

With prepared `data/processed/anymal_1_1_compslam/{est,gt}.tum`:

```bash
python scripts/reproduce_all.py --seed 0
```

See [docs/methodology.md](docs/methodology.md). Manual CI dispatch also runs
slow tests plus reproduce-all when data is present.

### Testing philosophy

1. **Analytic answers.** Every non-trivial estimator-style function (alignment,
   fusion, protection-level math, separation statistics, metric formulae) needs
   at least one test with a **known analytic or hand-computed answer**, not only
   a smoke check that "it returns a float".
2. **Goldens are contracts.** Do not change values in
   `tests/golden/expected_results.json` (or other frozen headline numbers)
   without explaining **why the underlying result moved** in the pull request
   (algorithm change, config default change, data revision, or bugfix). Silent
   golden edits will be rejected.
3. Prefer focused unit tests next to the behaviour you changed; add a slow
   regression only when a publishable number is at stake.

## Code style

- **Ruff** (including import sorting and the bandit `S` family): must pass on
  `src/`, `tests/`, and `scripts/`. Config lives in `pyproject.toml`.
- **Mypy** `--strict` on the **public** package surface (`mypy` with the repo
  config). Internals under `_internal` are excluded from that gate.
- Match existing naming, dataclasses, and error types (`CairnInputError` for
  partner-facing input failures).
- No credentials, `.env` files, or licensed datasets in commits.

```bash
ruff check src tests scripts
mypy
```

## Pull requests and CI

1. Branch from an up-to-date `main`.
2. Keep the change focused; update docs when behaviour or config changes.
3. Open a PR using the template checklist.
4. CI on every push/PR (see `.github/workflows/`):
   - `ci.yml`: ruff, mypy (public surface), fast pytest, pre-commit secret/hygiene scan
   - `security.yml`: pip-audit and osv-scanner on the lockfile
5. Required checks and linear history are expected once repository rulesets are
   enabled ([docs/repository-settings.md](docs/repository-settings.md)). Solo
   maintainers may merge without a second approving review until a second
   maintainer exists.

## How to add a new detector

Most likely contribution path.

1. **Implement** under `src/cairn/_internal/detectors/` (or baselines if it is a
   covariance-only method). Take validated `PoseEstimate` inputs and produce an
   `IntegrityReport` (and optional diagnostic records).
2. **Wire** a new `monitor.mode` value through `CairnConfig` /
   `configs/default.yaml` and `CairnMonitor._assess` so callers can select it.
3. **Export** only what belongs in the public API (usually nothing new beyond
   config). Keep algorithms internal until there is a stable interface.
4. **Test** with analytic cases (known separations, known chi-squared thresholds,
   single-source degradation, filthy inputs). Add a focused `tests/test_*.py`.
5. **Document** the mode in `docs/configuration.md` and, if it changes product
   claims, `docs/concepts.md` / `docs/limitations.md`.
6. **Do not** bump golden headline numbers unless the detector changes a
   published result; if it does, explain the move in the PR.

Sketch:

```python
# src/cairn/_internal/detectors/my_detector.py
from cairn.types import IntegrityReport, IntegrityState, PoseEstimate

class MyDetector:
    def assess(self, timestamp: float, estimates: dict[str, PoseEstimate]) -> IntegrityReport:
        ...
```

Then branch in `CairnMonitor._assess` on `mode == "my_mode"`.

## Security and dependencies

- Report vulnerabilities privately ([SECURITY.md](SECURITY.md)).
- Do not commit secrets; regenerate `.secrets.baseline` only when intentional
  false positives appear (see comments in `.pre-commit-config.yaml`).
- `pip-audit -r requirements.lock.txt` should stay clean, or document an ignore
  in `docs/dependency-audit.md`.
- Numpy must stay on 1.x (`numpy==1.26.4`). Scipy stays on `1.17.1` (1.18.0
  breaks tests; both pins sit with gtsam's numpy 1.x ABI). Dependabot ignores
  numpy major bumps and scipy minor/major bumps.

### Which dependency file is authoritative

Do not edit the three files independently.

| File | Role | When to use |
| --- | --- | --- |
| `pyproject.toml` | **Declared pins** (runtime `[project.dependencies]`, optional `[dev]`) | Change versions here first |
| `requirements.txt` | **Hash-pinned runtime lock** generated from `pyproject.toml` | `pip install --require-hashes -r requirements.txt`; supply-chain / Scorecard |
| `requirements.lock.txt` | **Evaluation freeze** (evo, gtsam, pytest, and the same runtime pins) | `pip-audit`, SBOM, full data-tooling reproduce |

After any runtime pin change in `pyproject.toml`:

```bash
pip install pip-tools
pip-compile --generate-hashes --output-file=requirements.txt pyproject.toml
```

Then align overlapping names in `requirements.lock.txt` (numpy, scipy,
matplotlib, PyYAML, and their transitives) to the compiled versions. Extra
lock-only packages (evo, gtsam, …) are not in `requirements.txt` on purpose.

## Conduct

Be kind. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
