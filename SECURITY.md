# Security policy

## Supported versions

| Version | Security fixes |
| --- | --- |
| 0.1.x | Yes |
| older than 0.1 | No |

## How to report a vulnerability

**Prefer private disclosure.** Do not open a public GitHub issue or pull request for a security finding.

1. **Primary:** use GitHub private vulnerability reporting:
   https://github.com/GautamTalksDev/Cairn/security/advisories/new
2. **Fallback:** email `developwith.gt@gmail.com` with the subject line `Cairn security: <short title>`.

## What to include

A useful report includes:

- Affected Cairn version (and install method if relevant)
- Clear reproduction steps
- Impact (what an attacker can do, and under what assumptions)
- Environment (OS, Python version, relevant dependencies)

## What to expect

As a small team we commit to:

- **Acknowledgement** within **72 hours** of a private report
- **Initial assessment** within **7 days** (severity, whether we can reproduce, whether we will fix)
- **Coordinated disclosure** once a fix exists (or once we agree no fix is warranted)

We do not promise a fixed patch date beyond that assessment.

## Out of scope

- Public GitHub issues or pull requests filed for security matters
- Vulnerabilities in third-party dependencies (report those upstream; tell us if Cairn’s use amplifies impact)
- Attacks that require physical access to a user’s machine
- Social engineering of maintainers or users

## Threat model

**What Cairn is.** Cairn is a library and CLI that consumes pose estimates (and related navigation inputs) and produces integrity assessments: protection levels, integrity state, and related metrics. Callers integrate it into a larger navigation stack.

**What Cairn is responsible for.**

- Correctness of its own algorithms and APIs under the documented input conventions
- Safe handling of malformed inputs at public API boundaries (reject or mark untrusted where implemented)
- Keeping the published Cairn codebase free of accidental credential leakage and supply-chain hygiene issues in this repository

**What Cairn is not responsible for.**

- **Authenticating or attesting its inputs.** Cairn does **not** authenticate pose estimates, covariances, timestamps, or source identities. A caller that feeds forged, replayed, or manipulated pose data will receive an integrity assessment of that forged data. In any adversarial deployment, upstream authentication, transport security, and trust in estimators are the caller’s responsibility; Cairn cannot detect a globally consistent lie that looks like valid input.
- Securing the host, robot, network, or estimator that produces poses
- Certifying field safety (SIL/ASIL) or guaranteeing missed-detection bounds in deployment
- Third-party datasets (e.g. CERBERUS) used only for evaluation

## Safe harbour

Good-faith security research on Cairn is welcome. We will not pursue legal action against researchers who:

- Report privately through the channels above
- Avoid privacy violations, service degradation, and access beyond what is needed to demonstrate the issue
- Give us a reasonable chance to fix before public disclosure

Public dump of an unpatched vulnerability, or research that harms users or systems, is outside this safe harbour.
