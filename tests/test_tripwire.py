import pytest

from infrastructure.automation.tripwire import (
    TripwireCrossingDetector,
    segments_intersect,
    side_of_line,
)

# dom_c1: the gate is above the line (smaller y), the parking spots below it.
LINE = [[200, 500], [1000, 500]]


def feed(detector, points_per_step, start=0.0, step=1.0):
    crossings = []
    for i, points in enumerate(points_per_step):
        crossings += detector.update(points, start + i * step)
    return crossings


def test_gate_side_is_the_trigger_side():
    assert side_of_line((600, 300), LINE) > 0
    assert side_of_line((600, 700), LINE) < 0


def test_segments_intersect():
    assert segments_intersect((600, 600), (600, 400), (200, 500), (1000, 500))
    assert not segments_intersect((1200, 600), (1200, 400), (200, 500), (1000, 500))


def test_departure_towards_gate_fires_once():
    # Logged 3 July 12:54: car leaves the parking spot and drives to the gate.
    detector = TripwireCrossingDetector(LINE)
    crossings = feed(detector, [[(424, 614)], [(539, 425)], [(672, 319)], [(798, 245)]])
    assert crossings == [((424.0, 614.0), (539.0, 425.0))]


def test_arrival_through_gate_does_not_fire():
    # Logged 3 July 13:02: car comes in through the open gate.
    detector = TripwireCrossingDetector(LINE)
    assert feed(detector, [[(509, 219)], [(512, 254)], [(608, 294)], [(718, 454)], [(760, 560)]]) == []


def test_arrival_fires_when_direction_is_reversed():
    detector = TripwireCrossingDetector(LINE, direction="to_approach_side")
    assert len(feed(detector, [[(718, 454)], [(760, 560)]])) == 1


def test_car_parked_on_gate_side_never_fires():
    # Logged 3 July 11:43-11:44: a car stood by the gate for over a minute.
    detector = TripwireCrossingDetector(LINE)
    assert feed(detector, [[(605, 274)]] * 80) == []


def test_crossing_outside_the_segment_does_not_fire():
    detector = TripwireCrossingDetector(LINE)
    assert feed(detector, [[(1400, 620)], [(1450, 420)]]) == []


def test_cached_detections_repeated_between_ai_runs_are_ignored():
    detector = TripwireCrossingDetector(LINE)
    steps = [[(600, 560)]] * 5 + [[(600, 450)]] * 5
    assert len(feed(detector, steps, step=0.04)) == 1


def test_old_observation_is_not_matched():
    detector = TripwireCrossingDetector(LINE, max_gap_seconds=3.0)
    assert feed(detector, [[(600, 560)], [(600, 450)]], step=10.0) == []


def test_jump_between_two_cars_is_not_a_crossing():
    detector = TripwireCrossingDetector(LINE, max_jump_px=300)
    # one car parked below the line, another one appears far away above it
    assert feed(detector, [[(300, 700)], [(300, 700), (950, 300)]]) == []


def test_unknown_direction_is_rejected():
    with pytest.raises(ValueError):
        TripwireCrossingDetector(LINE, direction="sideways")
