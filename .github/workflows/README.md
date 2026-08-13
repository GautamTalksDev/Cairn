# GitHub Actions workflows

Hardening rules (CP-24): top-level `permissions: contents: read`, actions pinned
to full commit SHAs, no `pull_request_target`, no untrusted `${{ }}` in `run:`
shells, `persist-credentials: false` on checkout, job `timeout-minutes`, and
concurrency cancel-in-progress. Fork pull requests never receive write scopes
or repository secrets.

| Workflow | Triggers | Purpose |
| --- | --- | --- |
| [`ci.yml`](ci.yml) | `push`, `pull_request`, manual `workflow_dispatch` | Lint (ruff), public-surface mypy, fast pytest, pre-commit secret/hygiene scan; optional slow tests + `reproduce_all.py` on manual dispatch |
| [`security.yml`](security.yml) | `push` to `main`, `pull_request`, weekly schedule, `release` | `pip-audit` + `osv-scanner` on `requirements.lock.txt`; SARIF upload to the Security tab (same-repo only); CycloneDX `sbom.json` artifact on release |
| [`scorecard.yml`](scorecard.yml) | `push` to `main`, weekly schedule | OpenSSF Scorecard analysis; publishes results for the public badge; SARIF → Security tab (skipped on forks) |
| [`release.yml`](release.yml) | push of tags `v*` | Build sdist/wheel + SBOM, Sigstore provenance attestations, GitHub Release assets (`release` environment) |

## Permissions

### `ci.yml`

| Job | Permissions | Why |
| --- | --- | --- |
| *(workflow default)* | `contents: read` | Deny all other scopes by default |
| `test` | `contents: read` | Checkout + install + test only |
| `security-scan` | `contents: read` | Same; pre-commit does not push |
| `release-hygiene` | `contents: read` | Manual dispatch only; never on `pull_request` |

No job requests write scopes. No secrets are referenced.

### `security.yml`

| Job | Permissions | Why |
| --- | --- | --- |
| *(workflow default)* | `contents: read` | Deny write by default |
| `dependency-audit` | `contents: read` | Runs for fork PRs safely; uploads SARIF/JSON as artifacts only |
| `osv-sarif-upload` | `contents: read`, `security-events: write` | Uploads SARIF to code scanning; **skipped for fork PRs** (`head.repo.full_name == github.repository` gate) |
| `sbom` | `contents: read` | Release-only; produces `sbom.json` artifact |

### `scorecard.yml`

| Job | Permissions | Why |
| --- | --- | --- |
| *(workflow default)* | `read-all` | Scorecard’s documented default; no workflow-level write |
| `analysis` | `security-events: write`, `id-token: write`, `contents: read`, `actions: read` | SARIF upload + OIDC for `publish_results: true` (badge / public dataset). Job is skipped when `github.event.repository.fork == true` |

Only the Scorecard job may hold `id-token: write` among analysis workflows (publish_results restriction).

### `release.yml`

| Job | Permissions | Why |
| --- | --- | --- |
| *(workflow default)* | `contents: read` | Deny write by default |
| `release` | `contents: write`, `id-token: write`, `attestations: write` | Create GitHub Release + Sigstore attestations via OIDC (no long-lived publish token). Gated on environment `release`. |

## Triggers we refuse

- **`pull_request_target`**: not used. Fork PR code runs only under `pull_request` with a read-only token profile and no write job scopes.
- **Shell interpolation of PR titles / branch names / issue bodies**: not used. Matrix Python versions are passed via action `with:` inputs, not `run:` string expansion of attacker-controlled event fields.

## Action pins

Third-party actions are pinned to commit SHAs with the human-readable tag in a trailing comment, for example:

```yaml
uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
```

Bump pins deliberately (review the release diff), never via a floating mutable tag.
