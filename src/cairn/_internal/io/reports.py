"""Integrity report file I/O.

Whitespace-separated format (``#`` comments ignored)::

    timestamp pl_h pl_v state time_since_corroboration reason...

``state`` is one of ``nominal``, ``degraded``, ``do_not_trust``.
``reason`` is the remainder of the line (may contain spaces).
"""

from __future__ import annotations

__all__ = [
    "load_integrity_reports",
    "save_integrity_reports",
]


from collections.abc import Sequence
from pathlib import Path

from cairn.types import IntegrityReport, IntegrityState

_STATE_BY_VALUE = {s.value: s for s in IntegrityState}


def load_integrity_reports(path: str | Path) -> list[IntegrityReport]:
    """Load integrity reports from a whitespace-separated text file."""
    path = Path(path)
    reports: list[IntegrityReport] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"{path}:{line_no}: expected at least 5 fields "
                    "(timestamp pl_h pl_v state time_since_corroboration [reason...])"
                )
            timestamp = float(parts[0])
            pl_h = float(parts[1])
            pl_v = float(parts[2])
            state_tok = parts[3].lower()
            if state_tok not in _STATE_BY_VALUE:
                raise ValueError(
                    f"{path}:{line_no}: unknown state {parts[3]!r}; "
                    f"expected one of {sorted(_STATE_BY_VALUE)}"
                )
            t_corr = float(parts[4])
            reason = " ".join(parts[5:]) if len(parts) > 5 else ""
            reports.append(
                IntegrityReport(
                    timestamp=timestamp,
                    protection_level_horizontal=pl_h,
                    protection_level_vertical=pl_v,
                    state=_STATE_BY_VALUE[state_tok],
                    reason=reason,
                    time_since_corroboration=t_corr,
                )
            )
    if not reports:
        raise ValueError(f"{path}: no integrity report lines found")
    return reports


def save_integrity_reports(
    reports: Sequence[IntegrityReport],
    path: str | Path,
) -> None:
    """Write integrity reports in the whitespace-separated text format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# timestamp pl_h pl_v state time_since_corroboration reason",
    ]
    for r in reports:
        reason = r.reason.replace("\n", " ").strip()
        lines.append(
            f"{r.timestamp:.9f} {r.protection_level_horizontal:.9f} "
            f"{r.protection_level_vertical:.9f} {r.state.value} "
            f"{r.time_since_corroboration:.9f} {reason}".rstrip()
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
