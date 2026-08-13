# Repository settings checklist (CP-27)

Apply every item below on the GitHub repository **before** treating a release as
production-ready. This file is the source of truth; do not “almost” enable a
control.

**Repository:** `GautamTalksDev/Cairn` (adjust owner/name if the remote differs)
**Default branch:** `main`

Solo-maintainer policy (deliberate): require pull requests and passing checks on
`main`, but **do not** require an approving review until a second maintainer
exists. Otherwise you cannot merge your own work.

Mark each box when done in the live repo.

---

## 1. Branch protection / ruleset on `main`

Prefer a **ruleset** (readable by Scorecard with the default `GITHUB_TOKEN`).

Create: **Settings → Rules → Rulesets → New branch ruleset**

- [ ] Target: default branch `main` (or pattern `main`)
- [ ] Enforcement status: **Active**
- [ ] Restrict deletions: **on** (block branch deletion)
- [ ] Block force pushes: **on**
- [ ] Require a pull request before merging: **on**
  - [ ] Required approvals: **0** (solo maintainer  - revisit when a second maintainer joins)
  - [ ] Dismiss stale approvals: optional
  - [ ] Do **not** enable “Require review from Code Owners” until Code Owners exist
- [ ] Require status checks to pass: **on**
  - [ ] Require branches to be up to date: recommended **on**
  - [ ] Required checks (names must match Actions job names / check runs):
    - [ ] `test` (from `.github/workflows/ci.yml`)
    - [ ] `security-scan` (from `.github/workflows/ci.yml`)
    - [ ] `dependency-audit` (from `.github/workflows/security.yml`)  - optional if too slow for every PR
- [ ] Require linear history: **on** (no merge commits; squash or rebase)
- [ ] Do **not** allow bypassing the ruleset for ordinary collaborators (keep
      bypass list empty or admin-only with a recorded reason)

Equivalent classic branch protection is acceptable if rulesets are unavailable;
mirror the same constraints.

### CLI sketch (ruleset via API)

After `gh auth login` and `gh repo set-default`:

```bash
# Inspect current rulesets
gh api repos/GautamTalksDev/Cairn/rulesets

# Create/update is UI-preferred for first setup; verify:
gh api repos/GautamTalksDev/Cairn/rulesets --jq '.[].name'
```

---

## 2. Tag creation restricted to `v*`

Create: **Settings → Rules → Rulesets → New tag ruleset**

- [ ] Target tags: pattern `v*`
- [ ] Restrict creations: **on**  - only maintainers / release role may create
      version tags
- [ ] Block force updates / deletion of matching tags: **on** where available
- [ ] Document who may cut releases in [`RELEASING.md`](../RELEASING.md)

---

## 3. Actions default permissions (repository-wide)

**Settings → Actions → General → Workflow permissions**

- [ ] Default: **Read repository contents and packages permissions** (read-only)
- [ ] **Do not** allow GitHub Actions to create and approve pull requests
      (unless a documented bot workflow needs it later)
- [ ] Workflows may still request additional scopes per job (as in
      `scorecard.yml` / `release.yml`); the repository default stays read-only

---

## 4. Security product features

**Settings → Code security** (names vary slightly by plan)

- [ ] **Private vulnerability reporting**: Enabled
      (required by [`SECURITY.md`](../SECURITY.md); link must work before publish)
- [ ] **Dependabot alerts**: Enabled
- [ ] **Dependabot security updates**: Enabled
      (version updates already configured in `.github/dependabot.yml`)
- [ ] **Secret scanning**: Enabled
- [ ] **Push protection** (secret scanning): Enabled

```bash
# Private vulnerability reporting
gh api -X PUT repos/GautamTalksDev/Cairn/private-vulnerability-reporting

# Remaining toggles are primarily UI / org-policy controlled; confirm in Settings.
```

---

## 5. Environments

**Settings → Environments → New environment: `release`**

- [ ] Environment name: `release`
- [ ] Required reviewers: optional for solo maintainer (empty is OK); add when a
      second person can approve publishes
- [ ] Deployment branches/tags: limit to tags matching `v*` (or default branch
      only if you cut releases another way)
- [ ] No long-lived secrets stored here (no PyPI API token)

Used by `.github/workflows/release.yml`.

---

## 6. PyPI Trusted Publishing (next step  - not enabled yet)

Publishing to PyPI is **not decided**. When it is:

1. Create the PyPI project (or claim the name).
2. PyPI → **Publishing → Trusted publishers → Add a new pending publisher**:
   - Owner: `GautamTalksDev`
   - Repository: `Cairn`
   - Workflow: `release.yml`
   - Environment: `release`
3. Enable the `publish-pypi` job documented in [`RELEASING.md`](../RELEASING.md)
   (OIDC only: `permissions.id-token: write`, **no** `TWINE_PASSWORD` /
   API token).
4. Verify the uploaded files have Sigstore provenance / attestations.

Until then, releases use **GitHub Release artifacts + build provenance
attestations** only. **No long-lived publishing token should exist anywhere**
(not in Actions secrets, not in local env files, not in Dependabot).

---

## 7. Post-apply verification

- [ ] Open a test PR: merging to `main` without a PR is blocked
- [ ] Required checks appear and must pass
- [ ] Force-push to `main` fails
- [ ] Deleting `main` fails
- [ ] Creating a random tag `test-tag` fails if outside allowed actors; `v0.0.0-test` follows tag rules
- [ ] https://github.com/GautamTalksDev/Cairn/security/advisories/new loads
- [ ] Actions → a workflow run shows default token is read-only unless a job
      requests more
- [ ] A `v*` tag release produces attestations verifiable with
      `gh attestation verify`

---

## Decision log

| Decision | Choice | Why |
| --- | --- | --- |
| Required PR approvals | **0** | Solo maintainer; avoid self-merge deadlock |
| Linear history | Required | Clean bisect / release tags |
| PyPI publish | Deferred | Provenance on GitHub Release first |
| Publishing credentials | OIDC / none | No long-lived tokens |
