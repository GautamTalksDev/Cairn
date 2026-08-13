# Releasing Cairn

End-to-end process for cutting a versioned release with verifiable provenance.
There is **no long-lived PyPI (or other) publishing token** in this project.

## Preconditions

Complete [`docs/repository-settings.md`](docs/repository-settings.md) (branch/tag
rulesets, read-only Actions defaults, secret scanning, private vulnerability
reporting, `release` environment).

CI on `main` is green (`ci.yml`, `security.yml`).

## 1. Version bump

1. Set the version in `pyproject.toml` (`[project].version`) and
   `src/cairn/__init__.py` (`__version__`) to the same value (e.g. `0.1.1`).
2. Keep SemVer: `MAJOR.MINOR.PATCH`.

## 2. Changelog

1. Update [`CHANGELOG.md`](CHANGELOG.md): move items under a new
   `## [X.Y.Z]  - YYYY-MM-DD` section (Keep a Changelog style).
2. Do not hand-edit [`RESULTS.md`](RESULTS.md); regenerate via scripts if
   headline numbers change.

## 3. Land on `main`

1. Open a pull request (required by the `main` ruleset).
2. Wait for required checks (`test`, `security-scan`, …).
3. Merge with a **linear** history (squash or rebase  - no merge commits).
4. **Do not** require a second approving review until there is a second
   maintainer (see repository settings decision log).

## 4. Tag

From an up-to-date `main`:

```bash
git checkout main
git pull
git tag -a "vX.Y.Z" -m "vX.Y.Z"
git push origin "vX.Y.Z"
```

Only authorized actors may create `v*` tags (tag ruleset). Annotated tags are
preferred.

## 5. CI release workflow

Pushing `v*` runs [`.github/workflows/release.yml`](.github/workflows/release.yml)
in the **`release`** environment:

1. Builds sdist + wheel (`python -m build`)
2. Writes CycloneDX `dist/sbom.json`
3. Mints **Sigstore build-provenance attestations** for those artifacts
   (`actions/attest-build-provenance`, OIDC  - short-lived, no stored secret)
4. Creates a **GitHub Release** and attaches `dist/*`

Watch the run under the Actions tab. Confirm the Release page lists the
artifacts.

## 6. Verification

```bash
# Download release assets, then:
gh attestation verify dist/cairn-X.Y.Z-py3-none-any.whl \
  --repo GautamTalksDev/Cairn

gh attestation verify dist/cairn-X.Y.Z.tar.gz \
  --repo GautamTalksDev/Cairn
```

Expect a successful verification linking the artifact digest to this
repository’s `release.yml` workflow run.

Also spot-check:

```bash
python -m venv /tmp/cairn-rel && /tmp/cairn-rel/bin/pip install dist/cairn-*.whl
/tmp/cairn-rel/bin/python -c "import cairn; print(cairn.__version__)"
/tmp/cairn-rel/bin/cairn --help
```

## 7. PyPI Trusted Publishing (next step  - optional)

Not enabled yet. When you decide to publish:

1. On PyPI, add a **Trusted Publisher** for this repo:
   - Workflow: `release.yml`
   - Environment: `release`
   - Owner/repo: `GautamTalksDev/Cairn`
2. Add a job (same `release` environment) with:

   ```yaml
   permissions:
     id-token: write   # OIDC only — never a PyPI API token
   steps:
     - uses: actions/download-artifact@...
     - name: Publish to PyPI
       uses: pypa/gh-action-pypi-publish@76f52bc884231f62b9a034ebfe128415bbaabdfc # v1.12.4
       with:
         packages-dir: dist/
   ```

3. Do **not** create or store a PyPI API token. Trusted Publishing issues a
   short-lived credential per run and attaches provenance to the upload.

## Checklist

- [ ] Version strings match across `pyproject.toml` and `__init__.py`
- [ ] `CHANGELOG.md` updated
- [ ] PR merged to `main` with required checks green
- [ ] Annotated `vX.Y.Z` tag pushed
- [ ] `Release` workflow green; assets on the GitHub Release
- [ ] `gh attestation verify` succeeds on wheel and sdist
- [ ] No PyPI/API tokens in repo secrets or local files
