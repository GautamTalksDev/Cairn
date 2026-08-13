# Limitations

This is the long form of the README limitations section. Read it as a hostile
reviewer would.

## Single sequence evidence is an anecdote until repeated

The underground story leans on one prepared CERBERUS CompSLAM run
(`anymal_1_1_compslam`) plus synthetic corruptions of that run. A single
trajectory can look persuasive and still be special: geometry, speed profile,
estimator tuning, and ground-truth quality all couple into MDR and timing.

Until the same protocol is shown across many sequences and sites, treat headline
metres and seconds as **illustrative**, not as a general underground guarantee.
Multi-sequence scripts exist; they do not replace a broad evidence base that is
not yet in hand.

## Synthetic faults show capability under a script, not nature

Bias, dropout, drift, and common-mode cases are injected into estimates. That
is useful: clean CompSLAM can sit under a performance ceiling where a static
covariance already looks fine, so you need failures on demand to stress
monitors.

What synthesis **does** demonstrate: under these parameterised corruptions, the
consistency monitor can alarm while naive covariance does not (drift timing),
and solution separation fails when the fault is applied to every source
(common mode).

What synthesis **does not** demonstrate: that real mines produce the same
signatures, rates, or time constants; that operators would see 0.02 m/s
along-track drift; or that field networks fail independently the way the
simulator assumes.

## Common mode: measured failure of the independence assumption

Solution separation checks disagreement. When every source carries the same
bias, separations stay small and the monitor can remain mostly nominal while
error grows.

Measured contrast in the common-mode experiment (synthetic; see
[`RESULTS.md`](../RESULTS.md)):

- Common-mode: MDR **73.2%**, post-onset alarm fraction **9.0%**
- Single-source control with the same injected magnitude: MDR **11.6%**,
  post-onset alarm fraction **97.4%**

This is not a footnote. Any deployment that shares maps, clocks, or tightly
coupled estimators must assume SS can be blind together. Source diversity is a
requirement, not a slogan.

## Availability cost of consistency monitoring

MDR is the safety-critical rate. Availability is still part of the product.
Consistency monitoring and PL inflation spend nominal time and can raise false
alarms relative to a naive baseline that is always `NOMINAL`.

A monitor that abstains three quarters of the time can post a low MDR and still
leave the robot without a usable integrity claim. Judge monitors on MDR **and**
availability (and PL size), together. The [`RESULTS.md`](../RESULTS.md) integrity
table already shows
degeneracy-aware inflation cutting MDR while spending availability and raising
FAR.

## No field validation

There is no partner deployment study, no sealed bag from a production robot,
and no operational sign-off behind these numbers. Lab reproduction from public
CERBERUS exports is not field validation.

## Non commercial data licence

Full CERBERUS data is In Copyright, Non-Commercial Use Permitted (InC-NC). That
blocks commercial redistribution of the dataset and constrains how vendors may
ship demos that embed those trajectories. The repository must not become a
channel for leaking full sequences. Evaluation that depends on InC-NC data is
not freely redeployable in every commercial context.

## Sensitivity to sigma, and the clean-data inconvenient truth

Because CERBERUS TUM files omit covariance, evaluation attaches a claimed
diagonal σ. MDR for the naive baseline moves strongly with that choice.

Sigma sweep on **clean** CompSLAM: MDR ≤ 0.001 near σ* ≈ **6.94 cm** with HPL ≈
**0.258 m** (0.74× ATE **0.3463 m**). On that sequence a well-tuned static
covariance is already competitive. The product case for always-on consistency
monitoring is weaker on clean data and stronger on independent-source faults. If
you only show clean MDR tables, you can fool yourself.

## Cairn does not authenticate its inputs

Cairn consumes pose estimates and covariances you provide. It does not
authenticate estimators, sign poses, or prove that two "sources" are physically
independent. Feed forged, replayed, or colluding inputs and you will get an
integrity assessment of forged data.

In any adversarial setting, transport security, estimator attestation, and
source provenance are upstream responsibilities. Cairn will not save you from a
consistent lie.

## Not a certified integrity monitor

No SIL/ASIL claim, no formal proof of missed-detection bounds, no warranty for
field use. Research scaffolding.

## What would change this document

Broaden sequences and sites; add real fault logs; measure common-mode on real
shared-map failures; run partner field trials; replace claimed σ with estimator
covariances; add authentication or provenance at the integration boundary. Until
then, keep claims inside what the methodology can defend.
