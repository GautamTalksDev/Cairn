## Summary

Briefly describe the change and why it belongs in Cairn.

## Scope

- [ ] This does **not** add a position estimator (SLAM / odometry / similar)
- [ ] No CERBERUS or other licensed full datasets are committed
- [ ] No secrets or credentials are included

## Tests

- [ ] Fast tests pass locally: `pytest -q -m "not slow"`
- [ ] New or changed estimator-style maths have a test with a known analytic answer
- [ ] Slow tests considered / run if goldens or headline numbers are affected

## Docs

- [ ] Docs updated if behaviour, config, or public API changed
- [ ] Limitations / methodology updated if a published claim changed

## Golden values

- [ ] No golden / frozen headline numbers changed, **or**
- [ ] Goldens changed and the PR explains **why the underlying number moved**

## Checklist

- [ ] `pre-commit run --all-files` passes (or CI equivalent)
- [ ] `ruff check src tests scripts` and `mypy` (public surface) are clean
- [ ] PR title is clear; related issue linked if any
