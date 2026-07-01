from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import face_recognition
import numpy as np


@dataclass(slots=True)
class KnownFace:
    person_id: str
    display_name: str
    encoding: np.ndarray
    source_path: str


@dataclass(slots=True)
class FaceMatch:
    person_id: str
    display_name: str
    distance: float
    confidence: float
    source_path: str


class FaceIdentityStore:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        known_faces_dir: str | Path = "data/faces/known",
        index_path: str | Path = "data/faces/known_faces.pkl",
        display_names_path: str | Path = "data/faces/display_names.json",
        tolerance: float = 0.48,
        auto_reload: bool = True,
        reload_check_interval_seconds: float = 2.0,
        debug: bool = False,
    ):
        self.known_faces_dir = Path(known_faces_dir)
        self.index_path = Path(index_path)
        self.display_names_path = Path(display_names_path)

        self.tolerance = float(tolerance)
        self.auto_reload = bool(auto_reload)
        self.reload_check_interval_seconds = float(reload_check_interval_seconds)
        self.debug = bool(debug)

        self._known_faces: list[KnownFace] = []
        self._last_index_mtime: float | None = None
        self._last_reload_check_at = 0.0

    @property
    def known_faces_count(self) -> int:
        return len(self._known_faces)

    def start(
        self,
        rebuild_if_missing: bool = True,
        rebuild_if_stale: bool = True,
    ) -> None:
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        should_rebuild = False

        if rebuild_if_missing and not self.index_path.exists():
            should_rebuild = self._has_known_face_images()

            if self.debug:
                print(
                    f"[FaceIdentityStore] index missing, "
                    f"has_images={should_rebuild}"
                )

        elif rebuild_if_stale and self._is_index_stale():
            should_rebuild = True

            if self.debug:
                print("[FaceIdentityStore] index is stale, rebuilding")

        if should_rebuild:
            self.rebuild_index_from_directory(
                model="hog",
                number_of_times_to_upsample=1,
                num_jitters=1,
            )

        self.load_index_if_exists()

        print(
            f"[FaceIdentityStore] ready "
            f"known_faces={self.known_faces_count} "
            f"index={self.index_path}"
        )

    def load_index_if_exists(self) -> None:
        if not self.index_path.exists():
            if self.debug:
                print(f"[FaceIdentityStore] index does not exist: {self.index_path}")

            self._known_faces = []
            self._last_index_mtime = None
            return

        try:
            with self.index_path.open("rb") as file:
                payload = pickle.load(file)

            known_faces: list[KnownFace] = []

            for item in payload.get("known_faces", []):
                known_faces.append(
                    KnownFace(
                        person_id=str(item["person_id"]),
                        display_name=str(item["display_name"]),
                        encoding=np.asarray(item["encoding"], dtype=np.float64),
                        source_path=str(item.get("source_path", "")),
                    )
                )

            self._known_faces = known_faces
            self._last_index_mtime = self.index_path.stat().st_mtime

            if self.debug:
                print(
                    f"[FaceIdentityStore] loaded {len(self._known_faces)} "
                    f"face encodings from {self.index_path}"
                )

        except Exception as exc:
            print(f"[FaceIdentityStore] failed to load index {self.index_path}: {exc}")
            self._known_faces = []
            self._last_index_mtime = None

    def rebuild_index_from_directory(
        self,
        model: str = "hog",
        number_of_times_to_upsample: int = 1,
        num_jitters: int = 1,
    ) -> int:
        display_names = self._load_display_names()

        known_faces: list[KnownFace] = []

        self.known_faces_dir.mkdir(parents=True, exist_ok=True)

        person_dirs = [
            path
            for path in self.known_faces_dir.iterdir()
            if path.is_dir()
        ]

        for person_dir in sorted(person_dirs):
            person_id = person_dir.name
            display_name = display_names.get(person_id, person_id)

            image_paths = [
                path
                for path in person_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS
            ]

            for image_path in sorted(image_paths):
                encoding = self._encode_largest_face_from_image(
                    image_path=image_path,
                    model=model,
                    number_of_times_to_upsample=number_of_times_to_upsample,
                    num_jitters=num_jitters,
                )

                if encoding is None:
                    print(f"[FaceIdentityStore] no face found: {image_path}")
                    continue

                known_faces.append(
                    KnownFace(
                        person_id=person_id,
                        display_name=display_name,
                        encoding=encoding,
                        source_path=str(image_path),
                    )
                )

                print(
                    f"[FaceIdentityStore] added face: "
                    f"person_id={person_id} "
                    f"display_name={display_name} "
                    f"file={image_path}"
                )

        self._save_index(known_faces)
        return len(known_faces)

    def match(
        self,
        face_encoding: np.ndarray,
    ) -> FaceMatch | None:
        self._maybe_reload()

        if not self._known_faces:
            return None

        known_encodings = [
            known_face.encoding
            for known_face in self._known_faces
        ]

        distances = face_recognition.face_distance(
            known_encodings,
            face_encoding,
        )

        if distances.size == 0:
            return None

        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        if best_distance > self.tolerance:
            return None

        known_face = self._known_faces[best_index]

        confidence = self._distance_to_confidence(
            distance=best_distance,
            tolerance=self.tolerance,
        )

        return FaceMatch(
            person_id=known_face.person_id,
            display_name=known_face.display_name,
            distance=best_distance,
            confidence=confidence,
            source_path=known_face.source_path,
        )

    def _encode_largest_face_from_image(
        self,
        image_path: Path,
        model: str,
        number_of_times_to_upsample: int,
        num_jitters: int,
    ) -> np.ndarray | None:
        try:
            image = face_recognition.load_image_file(str(image_path))
            image = np.ascontiguousarray(image)
        except Exception as exc:
            print(f"[FaceIdentityStore] failed to read {image_path}: {exc}")
            return None

        try:
            face_locations = face_recognition.face_locations(
                image,
                number_of_times_to_upsample=number_of_times_to_upsample,
                model=model,
            )
        except Exception as exc:
            print(f"[FaceIdentityStore] face detection failed {image_path}: {exc}")
            return None

        if not face_locations:
            return None

        largest_location = self._pick_largest_face(face_locations)

        try:
            encodings = face_recognition.face_encodings(
                image,
                known_face_locations=[largest_location],
                num_jitters=num_jitters,
                model="small",
            )
        except Exception as exc:
            print(f"[FaceIdentityStore] face encoding failed {image_path}: {exc}")
            return None

        if not encodings:
            return None

        return np.asarray(encodings[0], dtype=np.float64)

    def _save_index(self, known_faces: list[KnownFace]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "created_at": time.time(),
            "known_faces": [
                {
                    "person_id": known_face.person_id,
                    "display_name": known_face.display_name,
                    "encoding": known_face.encoding,
                    "source_path": known_face.source_path,
                }
                for known_face in known_faces
            ],
        }

        with self.index_path.open("wb") as file:
            pickle.dump(payload, file)

        print(
            f"[FaceIdentityStore] saved {len(known_faces)} encodings "
            f"to {self.index_path}"
        )

    def _load_display_names(self) -> dict[str, str]:
        if not self.display_names_path.exists():
            return {}

        try:
            with self.display_names_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            return {
                str(key): str(value)
                for key, value in payload.items()
            }

        except Exception as exc:
            print(f"[FaceIdentityStore] failed to read display names: {exc}")
            return {}

    def _maybe_reload(self) -> None:
        if not self.auto_reload:
            return

        now = time.monotonic()

        if now - self._last_reload_check_at < self.reload_check_interval_seconds:
            return

        self._last_reload_check_at = now

        if not self.index_path.exists():
            return

        try:
            mtime = self.index_path.stat().st_mtime
        except Exception:
            return

        if self._last_index_mtime is None or mtime != self._last_index_mtime:
            self.load_index_if_exists()

    def _has_known_face_images(self) -> bool:
        if not self.known_faces_dir.exists():
            return False

        for path in self.known_faces_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS:
                return True

        return False

    def _is_index_stale(self) -> bool:
        if not self.index_path.exists():
            return self._has_known_face_images()

        try:
            index_mtime = self.index_path.stat().st_mtime
        except Exception:
            return False

        latest_source_mtime = self._latest_source_mtime()

        if latest_source_mtime is None:
            return False

        return latest_source_mtime > index_mtime

    def _latest_source_mtime(self) -> float | None:
        latest: float | None = None

        if self.known_faces_dir.exists():
            for path in self.known_faces_dir.rglob("*"):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue

                try:
                    mtime = path.stat().st_mtime
                except Exception:
                    continue

                latest = mtime if latest is None else max(latest, mtime)

        if self.display_names_path.exists():
            try:
                display_names_mtime = self.display_names_path.stat().st_mtime
                latest = (
                    display_names_mtime
                    if latest is None
                    else max(latest, display_names_mtime)
                )
            except Exception:
                pass

        return latest

    def _pick_largest_face(
        self,
        face_locations: list[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, int]:
        def area(location: tuple[int, int, int, int]) -> int:
            top, right, bottom, left = location
            return max(0, right - left) * max(0, bottom - top)

        return max(face_locations, key=area)

    def _distance_to_confidence(
        self,
        distance: float,
        tolerance: float,
    ) -> float:
        if tolerance <= 0:
            return 0.0

        confidence = 1.0 - (distance / tolerance)
        return max(0.0, min(1.0, confidence))