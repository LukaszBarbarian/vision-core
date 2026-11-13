from application.runtime.camera_stats import CameraStatsRegistry


def test_live_camera_reports_fps_and_resolution():
    stats = CameraStatsRegistry()
    for i in range(11):
        stats.record_frame("cam", 2560, 1440, timestamp=100.0 + i * 0.1)
    stats.record_inference("cam", "car_detection", 250.4)
    stats.record_inference("cam", "anpr_detection", 700.2)

    snap = stats.snapshot("cam", now=101.2)

    assert snap["status"] == "live"
    assert snap["fps"] == 10.0
    assert snap["resolution"] == "2560×1440"
    assert snap["inference_ms"] == {"anpr_detection": 700, "car_detection": 250}


def test_connected_camera_without_frames_is_stalled():
    stats = CameraStatsRegistry()
    stats.record_frame("cam", 1280, 720, timestamp=100.0)

    assert stats.snapshot("cam", now=110.0)["status"] == "stalled"


def test_disconnected_camera_is_offline_and_keeps_its_error():
    stats = CameraStatsRegistry()
    stats.record_connection("cam", connected=False, error="timeout")

    snap = stats.snapshot("cam", now=0.0)
    assert snap["status"] == "offline"
    assert snap["last_error"] == "timeout"
