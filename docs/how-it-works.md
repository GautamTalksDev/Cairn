# How Cairn works

Cairn does not estimate pose. At each epoch it takes pose estimates from one or
more named sources and decides whether the fused navigation solution should be
trusted for a configured integrity risk.

## Solution separation

When two or more independent sources are available, Cairn runs a
position-level solution-separation test (same family of idea as aviation RAIM).
It forms a full weighted least-squares solution from all sources, then forms
subset solutions that leave one source out. Large separations between the full
solution and a subset, relative to the separation covariance, indicate that the
left-out source is inconsistent with the others. The monitor raises protection
levels and can move the integrity state from `NOMINAL` to `DEGRADED` or
`DO_NOT_TRUST`. It also tracks how long it has been since sources last
corroborated each other.

Solution separation only sees **disagreement between sources**. If every source
is corrupted the same way (a **synthetic** common-mode fault in our experiments),
separations stay small and the monitor can remain `NOMINAL` while true error
grows. That limit is measured in the common-mode experiment (**73.2%** MDR;
see [`RESULTS.md`](../RESULTS.md)) and called out in the README limitations
section.

## What the API returns

Each update yields an integrity report with at least:

- horizontal and vertical protection levels (metres)
- integrity state (`NOMINAL`, `DEGRADED`, `DO_NOT_TRUST`)
- time since last corroboration (seconds), when applicable

See `examples/integrate.py` for a minimal caller loop and `configs/default.yaml`
for thresholds. Broader integrity concepts: [concepts.md](concepts.md).
