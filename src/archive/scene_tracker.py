import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SceneState:
    scene_id: str
    scene_no: int
    reference: np.ndarray
    last_check_ts: float
    scene_started_ts: float
    consecutive_changes: int = 0
    last_debug_ts: float = 0.0


class SceneTracker:
    """
    Wykrywa zmiane kadru po obrazie RTSP.

    To nie jest zapis detekcji. Ten tracker zapisuje tylko referencyjne klatki
    scen PTZ: pierwsza scena oraz kolejne sceny po duzej zmianie obrazu.
    Detekcje zapisuje ArchiveWriter, gdy DetectionWorker dolaczy snapshot_frame
    do eventu "detections".
    """

    def __init__(
        self,
        scenes_dir: str = "archive_logs/scenes",
        check_interval_sec: float = 0.5,
        min_scene_duration_sec: float = 3.0,
        mean_threshold: float = 18.0,
        changed_tiles_threshold: float = 0.45,
        tile_threshold: float = 22.0,
        consecutive_required: int = 2,
        reference_blend_alpha: float = 0.02,
        debug: bool = False,
        debug_interval_sec: float = 2.0,
    ):
        self.scenes_dir = Path(scenes_dir)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.check_interval_sec = check_interval_sec
        self.min_scene_duration_sec = min_scene_duration_sec
        self.mean_threshold = mean_threshold
        self.changed_tiles_threshold = changed_tiles_threshold
        self.tile_threshold = tile_threshold
        self.consecutive_required = consecutive_required
        self.reference_blend_alpha = reference_blend_alpha
        self.debug = debug
        self.debug_interval_sec = debug_interval_sec

        self._states: dict[str, SceneState] = {}

        print(f"[SceneTracker] scenes_dir={self.scenes_dir.resolve()}")

    def update(
        self,
        camera_id: str,
        frame,
        now: float | None = None,
    ) -> tuple[str, bool]:
        now = now if now is not None else time.time()

        if frame is None or getattr(frame, "size", 0) == 0:
            state = self._states.get(camera_id)
            if state is None:
                return f"{camera_id}_scene_0000", False
            return state.scene_id, False

        fingerprint = self._fingerprint(frame)
        state = self._states.get(camera_id)

        if state is None:
            scene_id = f"{camera_id}_scene_0001"
            self._states[camera_id] = SceneState(
                scene_id=scene_id,
                scene_no=1,
                reference=fingerprint,
                last_check_ts=now,
                scene_started_ts=now,
                consecutive_changes=0,
                last_debug_ts=now,
            )
            self._save_reference(camera_id, scene_id, frame)
            print(f"[SceneTracker] {camera_id}: initial scene {scene_id}")
            return scene_id, True

        if now - state.last_check_ts < self.check_interval_sec:
            return state.scene_id, False

        state.last_check_ts = now

        diff = cv2.absdiff(fingerprint, state.reference)
        mean_diff = float(diff.mean())
        changed_ratio = self._changed_tile_ratio(diff)

        looks_changed = (
            mean_diff >= self.mean_threshold
            and changed_ratio >= self.changed_tiles_threshold
        )

        if looks_changed:
            state.consecutive_changes += 1
        else:
            state.consecutive_changes = 0
            state.reference = cv2.addWeighted(
                state.reference,
                1.0 - self.reference_blend_alpha,
                fingerprint,
                self.reference_blend_alpha,
                0,
            )

        if self.debug and now - state.last_debug_ts >= self.debug_interval_sec:
            state.last_debug_ts = now
            print(
                f"[SceneTracker] {camera_id}: scene={state.scene_id} "
                f"mean_diff={mean_diff:.1f}/{self.mean_threshold:.1f} "
                f"changed_tiles={changed_ratio:.2f}/{self.changed_tiles_threshold:.2f} "
                f"consecutive={state.consecutive_changes}/{self.consecutive_required}"
            )

        enough_time_passed = now - state.scene_started_ts >= self.min_scene_duration_sec

        should_create_new_scene = (
            state.consecutive_changes >= self.consecutive_required
            and enough_time_passed
        )

        if not should_create_new_scene:
            return state.scene_id, False

        state.scene_no += 1
        state.scene_id = f"{camera_id}_scene_{state.scene_no:04d}"
        state.reference = fingerprint
        state.scene_started_ts = now
        state.consecutive_changes = 0
        state.last_debug_ts = now

        self._save_reference(camera_id, state.scene_id, frame)

        print(
            f"[SceneTracker] {camera_id}: scene changed -> {state.scene_id} "
            f"mean_diff={mean_diff:.1f} changed_tiles={changed_ratio:.2f}"
        )

        return state.scene_id, True

    def get_reference_path(self, camera_id: str, scene_id: str) -> Path:
        return self.scenes_dir / camera_id / f"{scene_id}.jpg"

    def _fingerprint(self, frame) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            gray,
            (160, 90),
            interpolation=cv2.INTER_AREA,
        )
        small = cv2.GaussianBlur(small, (7, 7), 0)
        return small

    def _changed_tile_ratio(
        self,
        diff: np.ndarray,
        tile_w: int = 16,
        tile_h: int = 9,
    ) -> float:
        h, w = diff.shape[:2]
        changed = 0
        total = 0

        for y in range(0, h, tile_h):
            for x in range(0, w, tile_w):
                tile = diff[y:y + tile_h, x:x + tile_w]
                if tile.size == 0:
                    continue

                total += 1

                if float(tile.mean()) >= self.tile_threshold:
                    changed += 1

        if total == 0:
            return 0.0

        return changed / total

    def _save_reference(
        self,
        camera_id: str,
        scene_id: str,
        frame,
    ) -> None:
        camera_dir = self.scenes_dir / camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)

        path = camera_dir / f"{scene_id}.jpg"

        ok = cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )

        if ok:
            print(f"[SceneTracker] saved scene reference: {path.resolve()}")
        else:
            print(f"[SceneTracker] FAILED to save scene reference: {path.resolve()}")