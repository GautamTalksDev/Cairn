# Dependency audit notes

`pip-audit` and `osv-scanner` run in `.github/workflows/security.yml` against
`requirements.lock.txt`.

## Current status

| Date | Tool | Result |
| --- | --- | --- |
| 2026-08-13 | `pip-audit==2.10.1` | No known vulnerabilities |
| 2026-08-13 | `osv-scanner` v2.5.0 | No issues found |

## Accepted findings

None. When a vulnerability must be temporarily accepted, add a row below with
advisory ID, package, reason, and expiry, and pass `--ignore-vuln <ID>` in the
workflow only for that ID.

| Advisory | Package | Reason | Review by |
| --- | --- | --- | --- |
| (none) | | | |

## Numpy pin

Do not upgrade numpy to 2.x via Dependabot or by hand without a coordinated
migration: evo depends on numpy, and gtsam’s wheels target the numpy 1.26.4
ABI. See `.github/dependabot.yml`.
