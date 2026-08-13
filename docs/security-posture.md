# Security posture (OpenSSF Scorecard)

**Tool:** OpenSSF Scorecard CLI `v5.5.0` (`c395761`)
**Scan date:** 2026-08-13
**Method:** `scorecard --local=. --show-details` on a clean tree of the
repository contents (including Scorecard / Dependabot / hardened workflows;
excluding `.venv`). Checks that require the GitHub API
(Branch-Protection, Code-Review, CI-Tests, Contributors, Maintained,
Signed-Releases, CII-Best-Practices, …) were **not** scored locally; they are
covered as expected gaps below.

**Local aggregate score: 6.6 / 10**

Scorecard measures *process hygiene*, not whether Cairn’s navigation logic is
safe in the field. Treat the badge as a maintenance-discipline signal. Do not
oversell it  - see also `docs/dependency-audit.md` for advisory scanning.

---

## Full per-check output (local, `--show-details`)

```
RESULTS
-------
Aggregate score: 6.6 / 10

Check scores:
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
|  SCORE  |          NAME          |                             REASON                              |                                                      DETAILS                                                      |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | Binary-Artifacts       | no binaries found in the repo                                   |                                                                                                                   |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | Dangerous-Workflow     | no dangerous workflow patterns detected                         |                                                                                                                   |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | Dependency-Update-Tool | update tool detected                                            | Info: detected update tool: Dependabot: .github/dependabot.yml:1                                                  |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | Fuzzing                | project is not fuzzed                                           | Warn: no fuzzer integrations found                                                                                |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | License                | license file not detected                                       | Warn: project does not have a license file                                                                        |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ?       | Packaging              | packaging workflow not detected                                 | Warn: no GitHub/GitLab publishing workflow detected.                                                              |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 3 / 10  | Pinned-Dependencies    | dependency not pinned by hash detected; score normalized to 3 | Warn: pipCommand not pinned by hash: .github/workflows/ci.yml:45                                                  |
|         |                        |                                                                 | Warn: pipCommand not pinned by hash: .github/workflows/ci.yml:46                                                  |
|         |                        |                                                                 | Warn: pipCommand not pinned by hash: .github/workflows/ci.yml:80                                                  |
|         |                        |                                                                 | Warn: pipCommand not pinned by hash: .github/workflows/ci.yml:81                                                  |
|         |                        |                                                                 | Warn: pipCommand not pinned by hash: .github/workflows/ci.yml:110                                                 |
|         |                        |                                                                 | Warn: pipCommand not pinned by hash: .github/workflows/ci.yml:111                                                 |
|         |                        |                                                                 | Warn: pipCommand not pinned by hash: .github/workflows/security.yml:145                                           |
|         |                        |                                                                 | Warn: pipCommand not pinned by hash: .github/workflows/security.yml:146                                           |
|         |                        |                                                                 | Warn: pipCommand not pinned by hash: .github/workflows/security.yml:48                                            |
|         |                        |                                                                 | Warn: pipCommand not pinned by hash: .github/workflows/security.yml:49                                            |
|         |                        |                                                                 | Info:  18 out of  18 GitHub-owned GitHubAction dependencies pinned                                                |
|         |                        |                                                                 | Info:   1 out of   1 third-party GitHubAction dependencies pinned                                                 |
|         |                        |                                                                 | Info:   0 out of  10 pipCommand dependencies pinned                                                               |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | SAST                   | no SAST tool detected                                           | Warn: no pull requests merged into dev branch                                                                     |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | Security-Policy        | security policy file not detected                               | Warn: no security policy file detected                                                                            |
|         |                        |                                                                 | Warn: no security file to analyze                                                                                 |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | Token-Permissions      | GitHub workflow tokens follow principle of least privilege      | Info: jobLevel 'contents'/'actions' read on CI/security/scorecard jobs; topLevel read / read-all                  |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | Vulnerabilities        | 0 existing vulnerabilities detected                             |                                                                                                                   |
|---------|------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
```

Remediation links for each check live under
https://github.com/ossf/scorecard/blob/main/docs/checks.md

---

## Checks below 10  - meaning and fix path

### Fuzzing  - 0/10  - *will not satisfy soon*

**Meaning:** No OSS-Fuzz / ClusterFuzzLite / continuous fuzzer integration.
**Fix:** Add fuzz targets (e.g. TUM / report parsers, monitor input validation)
and wire them to OSS-Fuzz or CI fuzzing.
**Honest status:** Not planned for this research scaffolding. Unit tests are
not a substitute in Scorecard’s eyes.

### License  - previously 0/10

**Meaning:** Scorecard wants an OSI-recognized root `LICENSE` file (it does not
treat `pyproject.toml` text alone as enough).
**Fix applied:** root [`LICENSE`](../LICENSE) is Apache License 2.0;
`pyproject.toml` declares `Apache-2.0`. Keep CERBERUS dumps out of the tree.
**Honest status:** Done in-tree; Scorecard will reflect it after the file is on
the public default branch.

### Security-Policy  - previously 0/10

**Meaning:** Scorecard looks for a security policy file.
**Fix applied (CP-26):** root [`SECURITY.md`](../SECURITY.md) with private reporting
instructions, threat model, and safe harbour. Re-score after publish.

### SAST  - 0/10  - *addressable after GitHub*

**Meaning:** No CodeQL/Semgrep-style SAST on the default branch / merged PRs.
**Fix:** Enable CodeQL (or equivalent) on `push`/`pull_request`.
**Honest status:** CI runs ruff’s bandit `S` rules; Scorecard does not count
that as SAST. CodeQL is the usual fix.

### Pinned-Dependencies  - 3/10  - *partially addressable*

**Meaning:** Every GitHub Action is SHA-pinned (good, including
`ossf/scorecard-action`). Workflow `pip install` lines are not hash-pinned
(10 warnings).
**Fix:** Install CI tools from a hashed requirements file
(`pip install --require-hashes -r …`) or accept the score until that is worth
the friction.
**Honest status:** Acceptable near-term; Actions pinning was the high-value
part (CP-24).

### Packaging  - ?  - *deferred*

**Meaning:** No PyPI/GitHub release publishing workflow detected.
**Fix:** Add a trusted-publishing release workflow when wheels are published.
**Honest status:** Release SBOM exists; that does not satisfy Packaging.

---

## Checks at 10 (local)

| Check | Why |
| --- | --- |
| Binary-Artifacts | No committed binaries |
| Dangerous-Workflow | No `pull_request_target` / dangerous patterns flagged |
| Dependency-Update-Tool | Dependabot configured |
| Token-Permissions | Least-privilege workflow tokens |
| Vulnerabilities | No known OSV hits for declared deps at scan time |

---

## GitHub-only checks  - expected posture once public

| Check | Likely now | Note |
| --- | --- | --- |
| Code-Review | Low / 0 | **Will not satisfy** as a single-maintainer project until real multi-person review exists. |
| Branch-Protection | Low unless enabled | Turn on rulesets / protection on `main` (required checks, no force-push). |
| CI-Tests | Improves with PR usage | Fast CI already runs on `pull_request`. |
| Maintained | Activity-based | Follows recent default-branch commits. |
| Contributors | Low early | Few independent contributors/orgs. |
| Signed-Releases | 0 until signed | Signed tags / attested releases when shipping artifacts. |
| CII-Best-Practices | 0 | Separate OpenSSF Best Practices badge; not started. |

---

## Workflow and badge

- Workflow: `.github/workflows/scorecard.yml` (push to `main` + weekly;
  skipped on forks; `publish_results: true`; SARIF → Security tab).
- Badge in `README.md` points at `github.com/GautamTalksDev/Cairn`. It resolves
  after the first successful published run on the public default branch.
- Re-run locally: `scorecard --local=/path/to/clean/tree --show-details`
  (avoid scanning a working tree that contains `.venv`).
