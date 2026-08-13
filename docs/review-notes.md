# Documentation review notes (CP-32)

Hostile reading of the published documentation set after the consistency pass.
Keep this file; it is part of what we claim about our own evidence.

Scope reviewed: root `README.md`, `RESULTS.md`, `CONTRIBUTING.md`, `SECURITY.md`,
`RELEASING.md`, `CHANGELOG.md`, `docs/*.md` (concepts through limitations, plus
security/process notes), `examples/quickstart/README.md`, and the example
scripts referenced from getting-started.

Numbers were checked against the regenerated [`RESULTS.md`](../RESULTS.md)
(trajectory table from prepared CompSLAM; drift / common-mode / sigma from
`tests/golden/expected_results.json`).

---

## Claims stronger than the evidence

1. **README headline frames an underground product story.** Opening with silent
   failure of underground positioning, then a 12.1 s / 0.54 m detection, invites
   the reading that Cairn has been shown to work in mines. Evidence is one
   CERBERUS CompSLAM sequence plus **synthetic** faults. Field mines are not in
   the dataset.

2. **"Integrity monitoring" without a risk budget the reader can audit.** Docs
   correctly define protection levels and MDR, but rarely remind that the
   configured integrity risk (chi² / P_FA / PL confidence) is a modelling choice,
   not a measured P(HMI) from operations. A reader can treat 0.999 as an
   empirical guarantee.

3. **Solution separation described in RAIM language.** Aviation RAIM has
   standards, certified assumptions, and fleet evidence. Cairn's position-only
   WLS separation is in that *family of idea*, not a certified RAIM implementation.
   `docs/how-it-works.md` is careful; casual speech elsewhere can oversell.

4. **Degeneracy-aware inflation in the RESULTS table looks like a peer
   "method win".** MDR drops 15.08% → 10.59% with FAR rising to 9.29%. Internal
   notes and goldens already demote degeneracy on clean data (PL inflation /
   cowardice). The table alone can still be read as "degeneracy is better."
   Limitations and concepts now point at the availability cost; the table still
   needs that adjacent reading.

5. **Empirical SS false-alarm rate 5.05% ≈ configured 0.05.** That match is from
   a synthetic Monte Carlo under H0 with invented source stds, not from robot
   logs. It supports calibration of the detector under its own generative model.

6. **σ* ≈ 6.94 cm / HPL 0.258 m on clean data.** Accurate for this sequence. The
   stronger implication ("static covariance can already look fine") is true here
   and may not travel to harder sites. Stated in limitations; still easy to miss.

7. **"Research scaffolding, not certified"** is correct but easy to skip under a
   polished README with badges and a dramatic figure.

---

## Places a reader could reasonably feel misled

1. **RESULTS.md used to say "bundled quickstart" while carrying full-sequence
   numbers.** Fixed: the generator now prefers prepared CompSLAM when present,
   labels the section honestly, and embeds synthetic-fault / sigma headlines so
   README, concepts, methodology, and limitations share one table.

2. **README integrate sample showed five synthetic epochs** after
   `examples/integrate.py` switched to the quickstart TUM pair. Fixed expected
   output and wording.

3. **Rounding drift:** "about 12 s" vs locked **12.1 s**; "about 73%" vs
   **73.2%**; "≈6.9 cm / 0.26 m / ~0.7×" vs **6.94 cm / 0.258 m / 0.74×**.
   Docs now follow `RESULTS.md`.

4. **Placeholder `reports.txt` scoring MDR = 0.** Getting-started warns; a
   skimming reader can still think the quickstart score proves integrity
   performance. The RESULTS integrity table is a different experiment (naive /
   degeneracy on CompSLAM with claimed σ = 5 cm).

5. **Figure without reading the caption.** `drift_timing.png` looks like a field
   result. Caption and adjacent prose say synthetic; the image alone does not.

6. **Badges (CI, Scorecard) on a repo that may not yet be public / settings-complete.**
   Scorecard and Actions need a real remote and enabled settings
   (`docs/repository-settings.md`, `docs/security-posture.md`). Badges can imply
   a finished public posture.

7. **CERBERUS quickstart still in-tree under InC-NC.** Docs say do not commit
   full sequences; `SECURITY_AUDIT.md` still flags quickstart blobs as a
   publication blocker. A reader who only sees "bundled for offline demos" may
   miss the licence scrub requirement.

8. **Partner / integrator language** in older config and error paths (called out
   in the security audit) can imply a commercial programme that is not evidenced
   in these docs.

---

## Unstated assumptions (now partly documented, still easy to miss)

1. **Independence of sources.** Solution separation needs faults that do not hit
   every source the same way. Common-mode results quantify the failure; many
   stacks share maps, clocks, or tightly coupled estimators.

2. **Shared navigation frame at the API boundary.** Cairn does not transform
   frames. Wrong frame = wrong integrity claim.

3. **Claimed covariance when TUM has none.** Evaluation attaches diagonal σ
   (default 0.05 m). MDR for naive is dominated by that choice (sigma sweep).

4. **Umeyama alignment for offline scoring.** Removes gauge freedom for ATE /
   MDR tables. Live robots with known frames are a different problem.

5. **Duty-cycle detection metric (5 s, ≥50%, full post-onset window).** Headline
   12.1 s is not raw time-to-first-alarm. First-alarm can be FAR flicker.

6. **Horizontal metrics as the product story.** MDR/FAR tables use horizontal
   error and HPL. Vertical behaviour is reported but not the headline.

7. **No authentication.** Integrity of forged inputs is integrity of forged
   inputs (`SECURITY.md`, limitations). Upstream trust is assumed.

8. **Python 3.12 and numpy 1.x.** Runtime pins are real constraints; docs mention
   them unevenly outside CONTRIBUTING / dependency notes.

9. **Goldens and RESULTS synthetic sections move together.** Changing a locked
   number requires explaining why the underlying experiment moved
   (`CONTRIBUTING.md` testing philosophy).

---

## Consistency fixes applied in this pass

- Regenerated `RESULTS.md` with honest sequence labelling; added drift,
  common-mode, and sigma sections from goldens.
- `scripts/generate_results.py` prefers full CompSLAM when present; falls back
  to quickstart with a clear title.
- Aligned README, concepts, methodology, limitations, how-it-works to
  `RESULTS.md` numbers; marked synthetic faults wherever those numbers appear.
- Corrected README / getting-started integrate expectations.
- Terminology: protection level stays protection level (never renamed to a
  confidence-style radius).
- Removed em/en dashes from prose in user-facing markdown touched here (table
  rules and code fences unchanged).
- Verified internal links in the docs set; verified `cairn score`,
  `examples/integrate.py`, and the getting-started / api-reference snippets run.

---

## Residual risks (not fixed by docs alone)

- Public push still blocked on CERBERUS quickstart history scrub (`SECURITY_AUDIT.md`).
- No root `LICENSE` file yet (Scorecard License check).
- GitHub private vulnerability reporting and repository rulesets still require
  maintainer action after `gh auth`.
- Slow goldens need `data/processed/`; strangers with only quickstart cannot
  regenerate the full integrity table without preparing CERBERUS themselves.
