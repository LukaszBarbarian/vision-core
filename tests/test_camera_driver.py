from infrastructure.cameras.camera_driver import redact_credentials


def test_stream_credentials_are_hidden():
    message = "Immediate exit requested: 'rtsp://admin:secret%23@10.0.0.4:554/Streaming/Channels/101'"
    redacted = redact_credentials(message)

    assert "secret" not in redacted
    assert "admin" not in redacted
    assert "rtsp://***@10.0.0.4:554/Streaming/Channels/101" in redacted


def test_text_without_credentials_is_unchanged():
    assert redact_credentials("rtsp://10.0.0.4:554/live") == "rtsp://10.0.0.4:554/live"
