import av
import cv2
import time
from typing import Generator

class LiveStreamHandler:
    def __init__(self, camera_url: str):
        self.camera_url = camera_url

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        # Opcje FFMPEG dla RTSP/TCP
        options = {
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "max_delay": "500000",
        }

        container = av.open(self.camera_url, options=options)

        try:
            # Strumień wideo to zazwyczaj ścieżka nr 0
            stream = container.streams.video[0]
            
            for frame in container.decode(stream):
                # PyAV -> BGR (format dla OpenCV)
                img = frame.to_ndarray(format="bgr24")

                # Kompresja do JPEG
                ret, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

        except Exception as ex:
            print(f"Stream error: {ex}")
        finally:
            container.close()