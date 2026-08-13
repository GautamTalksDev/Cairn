"""Shared exception types for Cairn public APIs."""

from __future__ import annotations

__all__ = ["CairnInputError"]


class CairnInputError(ValueError):
    """Malformed navigation input rejected at a Cairn API boundary.

    Messages always name the offending ``timestamp`` and/or ``source`` when known
    so partners can locate bad samples without a debugger.

    Raised by
    --------
    :class:`~cairn.types.Pose` / :class:`~cairn.types.PoseEstimate` construction
    (via :class:`~cairn.monitor.CairnMonitor`), :meth:`CairnConfig.from_yaml`,
    and input validation inside the streaming monitor.
    """
