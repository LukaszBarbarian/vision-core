"""
Offline replay of archived detections through the tripwire rule logic.

Reads archive_logs/detections_long.csv + the archived snapshots of one camera
session, evaluates every car box against the camera's tripwire rule with the
same geometry the live AutomationRuleEngine uses, and renders an animated GIF:

    python tools/replay_tripwire.py --cam dom_c1 \
        --session dom_c1-20260703-111458 --from 114942 --to 115125 \
        --out docs/images/tripwire_replay.gif

Licence plates are blurred with the project's plate detector (--no-blur to disable).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infrastructure.automation.automation_orchestrator import AutomationRuleEngine  # noqa: E402
from infrastructure.config.settings import Settings  # noqa: E402

RED = (40, 40, 230)
GREEN = (80, 200, 80)
AMBER = (0, 190, 255)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)


def blur_plates(image: np.ndarray, plate_model) -> np.ndarray:
    if plate_model is None:
        return image
    boxes = []
    for imgsz in (640, 1280):  # plates are small in 1440p frames; scan at two scales
        result = plate_model(image, verbose=False, conf=0.3, imgsz=imgsz)[0]
        boxes.extend(result.boxes.xyxy.cpu().numpy().astype(int))
    for box in boxes:
        x1, y1, x2, y2 = box
        pad = 6
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)
        roi = image[y1:y2, x1:x2]
        if roi.size:
            image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), 18)
    return image


def draw_text(img, text, org, scale, color, thickness=2, bg=None, pad=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def render(image, rule, box, triggered, armed_since, t_rel, mask):
    out = image.copy()

    if mask:
        overlay = out.copy()
        cv2.fillPoly(overlay, [np.array(mask, np.int32)], (255, 150, 0))
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)

    (lx1, ly1), (lx2, ly2) = rule["line"]
    line_color = AMBER if triggered else RED
    cv2.line(out, (lx1, ly1), (lx2, ly2), line_color, 8, cv2.LINE_AA)
    draw_text(out, f"TRIPWIRE  {rule['name']}", (lx1, ly1 - 22), 1.1, WHITE, 3, bg=line_color)

    if box is not None:
        x1, y1, x2, y2 = box
        cx, by = (x1 + x2) // 2, y2
        cv2.rectangle(out, (x1, y1), (x2, y2), line_color if triggered else GREEN, 5)
        cv2.circle(out, (cx, by), 16, line_color if triggered else GREEN, -1)
        cv2.circle(out, (cx, by), 16, WHITE, 3)
        draw_text(out, "ground point", (cx + 26, by + 10), 0.9, WHITE, 2, bg=DARK, pad=6)

    # status panel
    panel_h = 150
    cv2.rectangle(out, (0, out.shape[0] - panel_h), (out.shape[1], out.shape[0]), DARK, -1)
    y = out.shape[0] - panel_h + 60
    if armed_since is None:
        draw_text(out, "car_detection  ->  ground point inside driveway", (40, y), 1.4, WHITE, 3)
        draw_text(out, "GATE: CLOSED", (40, y + 60), 1.4, GREEN, 3)
    else:
        draw_text(out, f"EVENT  {rule['trigger_event']}  (camera {rule['camera_id']})", (40, y), 1.4, AMBER, 3)
        draw_text(
            out,
            "GATE: ARMED  ->  multimodal rule listens for 'otworz brame' (8 s window)",
            (40, y + 60), 1.4, WHITE, 3,
        )
    draw_text(out, f"t = {t_rel:4.1f} s", (out.shape[1] - 300, y), 1.4, WHITE, 3)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", default="dom_c1")
    p.add_argument("--session", required=True)
    p.add_argument("--from", dest="frm", type=int, required=True)
    p.add_argument("--to", dest="to", type=int, required=True)
    p.add_argument("--csv", default="archive_logs/detections_long.csv")
    p.add_argument("--out", default="docs/images/tripwire_replay.gif")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--no-blur", action="store_true")
    args = p.parse_args()

    settings = Settings()
    config = settings.registry[args.cam]
    rule = next(r for r in config.camera.rules if r.get("type") == "tripwire")
    rule = {**rule, "camera_id": args.cam}
    engine = AutomationRuleEngine(frame_bus=None, event_bus=None, camera_config=config)

    df = pd.read_csv(ROOT / args.csv)
    df = df[
        (df.camera_id == args.cam)
        & (df.stream_session_id == args.session)
        & (df.frame_id.between(args.frm, args.to))
    ].copy()
    df["ts"] = pd.to_datetime(df.timestamp)

    plate_model = None
    if not args.no_blur:
        from ultralytics import YOLO
        plate_model = YOLO(str(ROOT / "license_plate_detector.pt"))

    frames, durations = [], []
    armed_since = None
    t0 = df.ts.min()

    for frame_id, group in df.groupby("frame_id", sort=True):
        snap = ROOT / Path(str(group.snapshot_path.iloc[0]).replace("\\", "/"))
        image = cv2.imread(str(snap))
        if image is None:
            continue

        cars = group[(group.label == "car") & (group.confidence >= 0.5)]
        box, triggered = None, False
        if not cars.empty:
            row = cars.sort_values("confidence").iloc[-1]
            box = tuple(int(v) for v in (row.x1, row.y1, row.x2, row.y2))
            point = ((box[0] + box[2]) // 2, box[3])
            triggered = engine._is_point_on_trigger_side(point=point, line=rule["line"])

        if triggered and armed_since is None:
            armed_since = frame_id

        image = blur_plates(image, plate_model)
        if plate_model is not None:
            # archived snapshots may carry ANPR boxes with the OCR text drawn above them
            for _, r in group[group.detection_type == "anpr_detection"].iterrows():
                x1, y1 = max(0, int(r.x1) - 10), max(0, int(r.y1) - 80)
                x2, y2 = min(image.shape[1], int(r.x2) + 10), min(image.shape[0], int(r.y2) + 10)
                image[y1:y2, x1:x2] = cv2.GaussianBlur(image[y1:y2, x1:x2], (0, 0), 18)
        t_rel = (group.ts.iloc[0] - t0).total_seconds()
        rendered = render(image, rule, box, triggered, armed_since, t_rel, config.camera.motion_mask_polygon)

        h = int(rendered.shape[0] * args.width / rendered.shape[1])
        small = cv2.resize(rendered, (args.width, h), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        durations.append(2200 if triggered and armed_since == frame_id else 900)

    if not frames:
        raise SystemExit("No snapshots found for the given range.")

    durations[-1] = 2500
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"Saved {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
