"""
Builds the analytics figures used in the README from archive_logs/detections_long.csv.

    python tools/make_readme_charts.py --session dom_c1-20260703-111458 \
        --background <snapshot.jpg>

Outputs docs/images/driveway_heatmap.png and docs/images/events_timeline.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from infrastructure.config.settings import Settings  # noqa: E402
from replay_tripwire import blur_plates  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {  # fixed categorical order
    "car_detection": ("#2a78d6", "Vehicle"),
    "person_detection": ("#eb6834", "Person"),
    "anpr_detection": ("#1baf7a", "Licence plate read"),
}
TRIPWIRE = "#e34948"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def heatmap(df, rule, background, out):
    cars = df[(df.detection_type == "car_detection") & (df.label == "car") & (df.confidence >= 0.5)]
    gx = (cars.x1 + cars.x2) / 2
    gy = cars.y2  # ground point = bottom-centre, the point the tripwire rule evaluates

    w, h = int(df.frame_width.iloc[0]), int(df.frame_height.iloc[0])
    fig, ax = plt.subplots(figsize=(12, 7.8), dpi=110)
    fig.patch.set_facecolor(SURFACE)

    bg = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    bg = (bg * 0.55 + 255 * 0.45).astype(np.uint8)  # wash out so the density reads first
    ax.imshow(bg, extent=(0, w, h, 0))

    sns.kdeplot(x=gx, y=gy, fill=True, cmap="Blues", thresh=0.08, levels=8, alpha=0.7, bw_adjust=0.6, ax=ax)
    ax.scatter(gx, gy, s=14, color="#184f95", alpha=0.55, linewidths=0)

    (lx1, ly1), (lx2, ly2) = rule["line"]
    ax.plot([lx1, lx2], [ly1, ly2], color=TRIPWIRE, linewidth=3)
    ax.annotate(
        f"tripwire  {rule['name']}  →  {rule['trigger_event']}",
        (lx1, ly1), xytext=(0, 10), textcoords="offset points",
        color=INK, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TRIPWIRE, lw=1.5),
    )
    ax.annotate("gate", (300, 330), color=INK, fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
    ax.annotate("parking spots", (700, 880), color=INK, fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))

    ax.set_xlim(0, min(w, 1800))  # crop to the driveway
    ax.set_ylim(min(h, 1100), 0)
    ax.set_axis_off()
    ax.set_title(
        f"Where vehicles are on the driveway — {len(cars)} vehicle ground points, 3 July",
        loc="left", color=INK, fontsize=14, fontweight="bold", pad=24,
    )
    ax.text(0, 1.01, "Ground point = bottom-centre of each car box, the point the tripwire rule evaluates",
            transform=ax.transAxes, color=INK_2, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def timeline(df, out):
    events = df.drop_duplicates(["frame_uid", "detection_type"]).copy()
    events["bin"] = events.ts.dt.floor("10min")
    counts = events.pivot_table(index="bin", columns="detection_type", values="frame_uid", aggfunc="count").fillna(0)
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="10min")
    counts = counts.reindex(full, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    width = pd.Timedelta(minutes=10) * 0.8
    bottom = np.zeros(len(counts))
    for key, (color, label) in SERIES.items():
        if key not in counts:
            continue
        vals = counts[key].to_numpy()
        ax.bar(counts.index, vals, width=width, bottom=bottom, color=color, label=label,
               edgecolor=SURFACE, linewidth=1.5, align="edge")
        bottom += vals

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.set_ylabel("Archived events per 10 min", color=INK_2, fontsize=11)
    frames = int(df.frame_id.max())
    ax.set_title(
        f"{frames:,} decoded frames → {len(events)} archived events (3 July, 11:15–13:51)",
        loc="left", color=INK, fontsize=14, fontweight="bold", pad=12,
    )
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", default="dom_c1")
    p.add_argument("--session", required=True)
    p.add_argument("--background", required=True, help="snapshot used as heat-map backdrop")
    p.add_argument("--csv", default="archive_logs/detections_long.csv")
    p.add_argument("--out-dir", default="docs/images")
    args = p.parse_args()

    config = Settings().registry[args.cam]
    rule = next(r for r in config.camera.rules if r.get("type") == "tripwire")

    df = pd.read_csv(ROOT / args.csv)
    df = df[(df.camera_id == args.cam) & (df.stream_session_id == args.session)].copy()
    df["ts"] = pd.to_datetime(df.timestamp)

    from ultralytics import YOLO
    background = blur_plates(cv2.imread(args.background), YOLO(str(ROOT / "license_plate_detector.pt")))

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap(df, rule, background, out_dir / "driveway_heatmap.png")
    timeline(df, out_dir / "events_timeline.png")
    print(f"Saved charts to {out_dir}")


if __name__ == "__main__":
    main()
