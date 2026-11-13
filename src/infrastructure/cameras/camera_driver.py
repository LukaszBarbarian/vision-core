# infrastructure/cameras/camera_driver.py

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from datetime import datetime

import av
import numpy as np
from av.audio.fifo import AudioFifo
from av.audio.resampler import AudioResampler

from domain.models.audio_chunk_context import AudioChunkContext
from domain.models.frame_context import FrameContext


class CameraDriver:
    def __init__(
        self,
        config,
        frame_bus,
        audio_bus=None,
        audio_sample_rate: int = 16000,
        audio_chunk_seconds: float = 1.0,
        debug_audio: bool = False,
        open_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 5.0,
    ):
        self.config = config
        self.frame_bus = frame_bus
        self.audio_bus = audio_bus

        self.audio_sample_rate = int(audio_sample_rate)
        self.audio_chunk_seconds = float(audio_chunk_seconds)
        self.debug_audio = bool(debug_audio)
        self.open_timeout_seconds = float(open_timeout_seconds)
        self.read_timeout_seconds = float(read_timeout_seconds)

        self.running = False

        # The RTSP reader runs in its own thread so network stalls and H.264
        # decoding never block the event loop. Frames are handed over through
        # a single latest-wins slot, so a busy loop sees fresh frames instead
        # of a growing backlog.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_lock = threading.Lock()
        self._pending_frame: FrameContext | None = None
        self._frame_flush_scheduled = False

    async def run(self):
        self.running = True
        self._loop = asyncio.get_running_loop()
        camera_id = self.config.camera.id

        try:
            while self.running:
                try:
                    await asyncio.to_thread(self._read_session)

                except asyncio.CancelledError:
                    raise

                except Exception as ex:
                    print(f"[{camera_id}] Driver error: {ex}")
                    traceback.print_exc()

                    if self.running:
                        print(f"[{camera_id}] Reconnecting in 2s...")
                        await asyncio.sleep(2)

                await asyncio.sleep(0.5)

        finally:
            # Lets the reader thread leave its loop at the next packet or read timeout.
            self.running = False

        print(f"[{camera_id}] Driver stopped.")

    def _read_session(self) -> None:
        """Blocking: opens the RTSP stream and reads it until it ends or the driver stops."""
        camera_id = self.config.camera.id
        container = None

        try:
            print(f"[{camera_id}] Opening RTSP stream...")

            container = av.open(
                self.config.camera.url,
                options={
                    "rtsp_transport": "tcp",
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                },
                timeout=(self.open_timeout_seconds, self.read_timeout_seconds),
            )

            video_stream = self._get_video_stream(container)
            audio_stream = self._get_audio_stream(container)

            streams = [video_stream]

            if audio_stream is not None:
                streams.append(audio_stream)

            stream_session_id = (
                f"{camera_id}-"
                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

            print(
                f"[{camera_id}] Stream opened. "
                f"session={stream_session_id}, "
                f"audio={'yes' if audio_stream is not None else 'no'}"
            )

            frame_id = 0
            audio_chunk_id = 0

            audio_resampler = None
            audio_fifo = None
            samples_per_chunk = int(
                self.audio_sample_rate * self.audio_chunk_seconds
            )

            if audio_stream is not None:
                audio_resampler = AudioResampler(
                    format="s16",
                    layout="mono",
                    rate=self.audio_sample_rate,
                )
                audio_fifo = AudioFifo()

            for packet in container.demux(*streams):
                if not self.running:
                    break

                if packet.stream == video_stream:
                    frame_id = self._handle_video_packet(
                        packet=packet,
                        camera_id=camera_id,
                        frame_id=frame_id,
                        stream_session_id=stream_session_id,
                    )

                elif audio_stream is not None and packet.stream == audio_stream:
                    audio_chunk_id, audio_fifo = self._handle_audio_packet(
                        packet=packet,
                        camera_id=camera_id,
                        audio_chunk_id=audio_chunk_id,
                        stream_session_id=stream_session_id,
                        audio_resampler=audio_resampler,
                        audio_fifo=audio_fifo,
                        samples_per_chunk=samples_per_chunk,
                    )

        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

    def _post_frame(self, ctx: FrameContext) -> None:
        """Called from the reader thread: keeps only the newest frame for the loop."""
        with self._frame_lock:
            self._pending_frame = ctx

            if self._frame_flush_scheduled:
                return

            self._frame_flush_scheduled = True

        self._loop.call_soon_threadsafe(self._flush_frame)

    def _flush_frame(self) -> None:
        """Runs on the event loop."""
        with self._frame_lock:
            ctx = self._pending_frame
            self._pending_frame = None
            self._frame_flush_scheduled = False

        if ctx is not None:
            self._loop.create_task(self.frame_bus.publish(ctx.camera_id, ctx))

    def _post_audio(self, audio_ctx: AudioChunkContext) -> None:
        """Called from the reader thread: audio chunks are small and must not be dropped."""
        asyncio.run_coroutine_threadsafe(
            self.audio_bus.publish(audio_ctx.camera_id, audio_ctx),
            self._loop,
        )

    def stop(self):
        self.running = False

    def _get_video_stream(self, container):
        if not container.streams.video:
            raise RuntimeError("RTSP stream does not contain video stream")

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        return stream

    def _get_audio_stream(self, container):
        audio_enabled = bool(getattr(self.config.camera, "audio_enabled", False))

        if not audio_enabled:
            return None

        if self.audio_bus is None:
            print(
                f"[{self.config.camera.id}] audio_enabled=True, "
                f"but audio_bus is None. Audio disabled."
            )
            return None

        if not container.streams.audio:
            print(f"[{self.config.camera.id}] No audio stream found.")
            return None

        stream = container.streams.audio[0]


        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"

        codec = getattr(stream.codec_context, "name", None)
        sample_rate = getattr(stream.codec_context, "sample_rate", None)
        channels = getattr(stream.codec_context, "channels", None)
        layout = getattr(stream.codec_context, "layout", None)
        format_name = getattr(stream.codec_context, "format", None)
        bit_rate = getattr(stream.codec_context, "bit_rate", None)

        print(
            f"[{self.config.camera.id}] Audio stream: "
            f"codec={codec}, sample_rate={sample_rate}, "
            f"channels={channels}, layout={layout}, "
            f"format={format_name}, bit_rate={bit_rate}"
        )

        return stream
        stream.thread_type = "AUTO"

        return stream

    def _handle_video_packet(
        self,
        packet,
        camera_id: str,
        frame_id: int,
        stream_session_id: str,
    ) -> int:
        try:
            frames = packet.decode()
        except Exception:
            print(f"[{camera_id}] Failed to decode video packet")
            traceback.print_exc()
            return frame_id

        for frame in frames:
            img = frame.to_ndarray(format="bgr24")

            ctx = FrameContext(
                camera_id=camera_id,
                frame_id=frame_id,
                frame_data=img,
                timestamp=time.time(),
                stream_session_id=stream_session_id,
            )

            self._post_frame(ctx)

            frame_id += 1

        return frame_id

    def _handle_audio_packet(
        self,
        packet,
        camera_id: str,
        audio_chunk_id: int,
        stream_session_id: str,
        audio_resampler: AudioResampler,
        audio_fifo: AudioFifo,
        samples_per_chunk: int,
    ) -> tuple[int, AudioFifo]:
        try:
            audio_frames = packet.decode()

        except av.error.InvalidDataError:
            # RTSP/EZVIZ/NVR potrafią czasem wysłać uszkodzony pakiet audio.
            # To nie jest powód do restartu kamery ani spamowania tracebackiem.
            if self.debug_audio:
                print(f"[{camera_id}] Dropped invalid audio packet")
            return audio_chunk_id, audio_fifo

        except Exception as ex:
            print(f"[{camera_id}] Failed to decode audio packet: {ex}")
            if self.debug_audio:
                traceback.print_exc()
            return audio_chunk_id, audio_fifo

        for audio_frame in audio_frames:
            try:
                resampled_frames = audio_resampler.resample(audio_frame)
            except Exception:
                print(f"[{camera_id}] Failed to resample audio frame")
                traceback.print_exc()
                continue

            if resampled_frames is None:
                continue

            if not isinstance(resampled_frames, (list, tuple)):
                resampled_frames = [resampled_frames]

            for resampled_frame in resampled_frames:
                try:
                    # WAŻNE:
                    # RTSP/NVR często daje audio frame z PTS np. 17920 zamiast 0.
                    # AudioFifo waliduje ciągłość PTS i potrafi się na tym wywalić.
                    #
                    # Ustawiamy tylko pts = None.
                    # NIE ustawiamy time_base = None, bo w Twojej wersji PyAV to powoduje:
                    # 'NoneType' object has no attribute 'numerator'
                    resampled_frame.pts = None

                    audio_fifo.write(resampled_frame)

                except ValueError as ex:
                    print(
                        f"[{camera_id}] Audio FIFO error: {ex}. "
                        f"Resetting audio fifo..."
                    )

                    audio_fifo = AudioFifo()

                    try:
                        resampled_frame.pts = None
                        audio_fifo.write(resampled_frame)
                    except Exception:
                        print(f"[{camera_id}] Failed to recover audio fifo")
                        traceback.print_exc()
                        continue

                except Exception:
                    print(f"[{camera_id}] Failed to write audio frame to fifo")
                    traceback.print_exc()
                    continue

            while audio_fifo.samples >= samples_per_chunk:
                chunk_frame = audio_fifo.read(samples_per_chunk)

                if chunk_frame is None:
                    break

                pcm = self._audio_frame_to_pcm_int16(chunk_frame)

                if pcm.size == 0:
                    continue

                timestamp_end = time.time()
                duration = float(len(pcm)) / float(self.audio_sample_rate)
                timestamp_start = timestamp_end - duration

                audio_ctx = AudioChunkContext(
                    camera_id=camera_id,
                    chunk_id=audio_chunk_id,
                    pcm=pcm,
                    sample_rate=self.audio_sample_rate,
                    timestamp_start=timestamp_start,
                    timestamp_end=timestamp_end,
                    stream_session_id=stream_session_id,
                )

                self._post_audio(audio_ctx)

                if self.debug_audio:
                    print(
                        f"[{camera_id}] audio chunk={audio_chunk_id} "
                        f"samples={len(pcm)} "
                        f"rate={self.audio_sample_rate} "
                        f"duration={duration:.2f}s"
                    )

                audio_chunk_id += 1

        return audio_chunk_id, audio_fifo

    def _audio_frame_to_pcm_int16(self, frame) -> np.ndarray:
        arr = frame.to_ndarray()
        arr = np.asarray(arr)

        # Po resamplingu do mono najczęściej będzie shape: (1, samples).
        # Zostawiamy jednak wariant defensywny.
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                arr = arr.reshape(-1)

        arr = arr.reshape(-1)

        if arr.dtype != np.int16:
            arr = arr.astype(np.int16)

        return arr