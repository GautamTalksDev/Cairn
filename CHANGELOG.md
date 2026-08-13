# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-13

### Added

- Public library API: `CairnMonitor`, `CairnConfig`, core types, and integrity metrics
- Streaming integrity monitor with consistency-based protection levels
- YAML configuration (`configs/default.yaml`) and structured diagnostics
- CLI harness: `cairn score` (summary, report, integrity plots)
- Partner integrate sample (`examples/integrate.py`) and quickstart trajectories
- Evaluation scripts (sigma sweep, fault injection, ROC, multiseq, benchmarks, reproduce-all)
- Packaging metadata and optional `dev` extras; Python ≥3.12; numpy pinned to 1.x

### Notes

- Full CERBERUS-backed headline numbers require prepared `data/processed/` (gitignored)
- Reproduce from this tag with locked runtime deps via `pip install .` then `python scripts/reproduce_all.py`
