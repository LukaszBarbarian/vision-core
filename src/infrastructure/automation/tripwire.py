"""
Tripwire geometry: decides when an object's ground point crosses a line segment.

A crossing is a move of the same object from the approach side of the line to
the trigger side, through the segment itself (not its infinite extension).
Detections carry no track id, so each new ground point is matched to the
nearest point seen recently, within ``max_gap_seconds`` and ``max_jump_px``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]


def side_of_line(point: Point, line) -> float:
    """Positive on the trigger side, negative on the approach side, 0 on the line."""
    (x1, y1), (x2, y2) = line
    px, py = point
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True when segment p1-p2 touches or crosses segment q1-q2."""

    def orient(a, b, c):
        value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (value > 0) - (value < 0)

    def on_segment(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    return (
        (o1 == 0 and on_segment(p1, p2, q1))
        or (o2 == 0 and on_segment(p1, p2, q2))
        or (o3 == 0 and on_segment(q1, q2, p1))
        or (o4 == 0 and on_segment(q1, q2, p2))
    )


@dataclass
class _Observation:
    point: Point
    timestamp: float


class TripwireCrossingDetector:
    """
    Stateful crossing test for one tripwire rule.

    ``direction``:
      * ``"to_trigger_side"`` - only approach side -> trigger side counts (default)
      * ``"to_approach_side"`` - only the reverse move counts
      * ``"any"`` - both directions count
    """

    def __init__(
        self,
        line,
        direction: str = "to_trigger_side",
        max_gap_seconds: float = 3.0,
        max_jump_px: float = 600.0,
    ):
        if direction not in {"to_trigger_side", "to_approach_side", "any"}:
            raise ValueError(f"Unknown tripwire direction: {direction!r}")

        (x1, y1), (x2, y2) = line
        self.line = ((float(x1), float(y1)), (float(x2), float(y2)))
        self.direction = direction
        self.max_gap_seconds = float(max_gap_seconds)
        self.max_jump_px = float(max_jump_px)

        self._history: list[_Observation] = []

    def update(self, points: list[Point], timestamp: float) -> list[tuple[Point, Point]]:
        """
        Feed the ground points of one AI result. Returns ``(previous, current)``
        pairs that crossed the line in the configured direction.

        The same points can be fed repeatedly (renderers re-publish cached
        detections between AI runs): a point identical to one already in the
        history is neither a move nor a new observation.
        """
        self._history = [
            obs for obs in self._history
            if timestamp - obs.timestamp <= self.max_gap_seconds
        ]

        known = {obs.point for obs in self._history}
        fresh = [tuple(map(float, p)) for p in points if tuple(map(float, p)) not in known]

        crossings = []

        for current in fresh:
            previous = self._nearest_previous(current)

            if previous is not None and self._is_crossing(previous.point, current):
                crossings.append((previous.point, current))

        self._history.extend(_Observation(point=p, timestamp=timestamp) for p in fresh)

        return crossings

    def reset(self) -> None:
        self._history.clear()

    def _nearest_previous(self, current: Point) -> _Observation | None:
        best, best_distance = None, None

        for obs in self._history:
            distance = math.dist(obs.point, current)

            if distance > self.max_jump_px:
                continue

            if best_distance is None or distance < best_distance:
                best, best_distance = obs, distance

        return best

    def _is_crossing(self, previous: Point, current: Point) -> bool:
        before = side_of_line(previous, self.line)
        after = side_of_line(current, self.line)

        if self.direction == "to_trigger_side":
            moved = before <= 0 < after
        elif self.direction == "to_approach_side":
            moved = before >= 0 > after
        else:
            moved = (before <= 0 < after) or (before >= 0 > after)

        return moved and segments_intersect(previous, current, *self.line)
