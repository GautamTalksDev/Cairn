"""ROS 2 bag trajectory loading (stub)."""

from __future__ import annotations

__all__ = [
    "load_ros2_bag",
]


from pathlib import Path

from cairn.types import Trajectory


def load_ros2_bag(path: str | Path, topic: str) -> Trajectory:
    """Load poses from a ROS 2 bag.

    Not implemented yet. Requires the ``rosbags`` package (and typically
    ROS message definitions for the pose topic).
    """
    raise NotImplementedError(
        "load_ros2_bag is not implemented yet; install and wire up the "
        f"'rosbags' dependency to read {path!s} topic {topic!r}"
    )
