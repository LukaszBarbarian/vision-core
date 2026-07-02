from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from infrastructure.ai_detection.face_identity_store import FaceIdentityStore


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--known-dir",
        default="data/faces/known",
    )

    parser.add_argument(
        "--index",
        default="data/faces/known_faces.pkl",
    )

    parser.add_argument(
        "--display-names",
        default="data/faces/display_names.json",
    )

    parser.add_argument(
        "--model",
        default="hog",
        choices=["hog", "cnn"],
    )

    parser.add_argument(
        "--upsample",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--jitters",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.48,
    )

    args = parser.parse_args()

    store = FaceIdentityStore(
        known_faces_dir=args.known_dir,
        index_path=args.index,
        display_names_path=args.display_names,
        tolerance=args.tolerance,
        debug=False,
    )

    count = store.rebuild_index_from_directory(
        model=args.model,
        number_of_times_to_upsample=args.upsample,
        num_jitters=args.jitters,
    )

    print(f"Done. Indexed {count} face encodings.")


if __name__ == "__main__":
    main()