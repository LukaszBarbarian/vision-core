from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from domain.models.camera_configuration import CameraConfiguration
from presentation.live_stream_handler import LiveStreamHandler

router = APIRouter(tags=["stream"])


@dataclass(slots=True)
class SnapshotInfo:
    camera_id: str
    path: Path
    rel_path: str
    filename: str
    session_id: str
    saved_at: datetime
    saved_at_text: str
    detection_type: str
    frame_id: str | None


def get_settings(request: Request):
    return request.app.state.settings


def get_audio_overlay_state(request: Request):
    audio_overlay_state = getattr(request.app.state, "audio_overlay_state", None)

    if audio_overlay_state is not None:
        return audio_overlay_state

    orchestrator = getattr(request.app.state, "orchestrator", None)

    if orchestrator is not None:
        return getattr(orchestrator, "audio_overlay_state", None)

    return None


def get_detector_provider(request: Request):
    detector_provider = getattr(request.app.state, "detector_provider", None)

    if detector_provider is not None:
        return detector_provider

    orchestrator = getattr(request.app.state, "orchestrator", None)

    if orchestrator is not None:
        detection_manager = getattr(orchestrator, "detection_manager", None)

        if detection_manager is not None:
            return getattr(detection_manager, "detector_provider", None)

    return None


def _escape(value) -> str:
    return html.escape(str(value or ""))


def _value_to_str(value) -> str:
    value = getattr(value, "value", value)
    return str(value)


def _format_fps(value: float) -> str:
    if value <= 0:
        return "-"

    if value.is_integer():
        return str(int(value))

    return f"{value:g}"


def _get_camera_config_or_404(settings, camera_id: str) -> CameraConfiguration:
    config: CameraConfiguration = settings.registry.get(camera_id)

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Camera with ID {camera_id} is missing from registry.",
        )

    return config


def _require_active_camera(config: CameraConfiguration):
    if not config.camera.active:
        raise HTTPException(
            status_code=400,
            detail="Camera is not active",
        )


def _get_detection_configs(settings, camera_id: str) -> list:
    detection_configs = getattr(settings, "detection_configs", {}) or {}
    return list(detection_configs.get(camera_id, []) or [])


def _get_allowed_detection_types(settings, camera_id: str) -> list[str]:
    specs = _get_detection_configs(settings, camera_id)
    result: list[str] = []

    for spec in specs:
        detection_type = getattr(spec, "detection_type", spec)
        result.append(_value_to_str(detection_type))

    return result


def _get_ai_fps(spec) -> float:
    value = getattr(spec, "ai_fps", 0)

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_requested_detections(
    requested: Optional[List[str]],
    allowed: list[str],
    default_all: bool,
) -> list[str]:
    if requested is None:
        return list(allowed) if default_all else []

    normalized_requested: list[str] = []

    for item in requested:
        if item is None:
            continue

        parts = [
            part.strip()
            for part in str(item).split(",")
            if part.strip()
        ]

        normalized_requested.extend(parts)

    if not normalized_requested:
        return list(allowed) if default_all else []

    lowered = {value.lower() for value in normalized_requested}

    if "none" in lowered:
        return []

    if "all" in lowered or "*" in lowered:
        return list(allowed)

    allowed_set = set(allowed)
    result: list[str] = []

    for detection_type in normalized_requested:
        if detection_type not in allowed_set:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Detection type '{detection_type}' is not configured. "
                    f"Configured options: {allowed}"
                ),
            )

        if detection_type not in result:
            result.append(detection_type)

    return result


def _build_url(
    path: str,
    query_items: list[tuple[str, str | int | float | bool]],
) -> str:
    return path + "?" + urlencode(query_items)


def _build_stream_url(
    camera_id: str,
    detections: list[str] | None = None,
    quality: int = 75,
    fps: float = 12.0,
    width: int | None = None,
) -> str:
    detections = detections or []

    query: list[tuple[str, str | int | float | bool]] = [
        ("cam", camera_id),
        ("quality", quality),
        ("fps", fps),
    ]

    for detection_type in detections:
        query.append(("detect", detection_type))

    if width:
        query.append(("width", width))

    return _build_url("/stream", query)


def _build_view_url(
    camera_id: str,
    detections: list[str] | None = None,
    raw: bool = False,
) -> str:
    detections = detections or []

    query: list[tuple[str, str | int | float | bool]] = [
        ("cam", camera_id),
        ("raw", "true" if raw else "false"),
    ]

    if not raw:
        if detections:
            for detection_type in detections:
                query.append(("detect", detection_type))
        else:
            query.append(("detect", "none"))

    return _build_url("/view", query)


def _snapshot_to_dict(snapshot: SnapshotInfo) -> dict:
    return {
        "camera_id": snapshot.camera_id,
        "filename": snapshot.filename,
        "rel_path": snapshot.rel_path,
        "image_url": _archive_image_url(snapshot),
        "session_id": snapshot.session_id,
        "saved_at": snapshot.saved_at.isoformat(timespec="seconds"),
        "saved_at_text": snapshot.saved_at_text,
        "detection_type": snapshot.detection_type,
        "frame_id": snapshot.frame_id,
    }

def _archive_image_url(snapshot: SnapshotInfo) -> str:
    return "/archive-image?" + urlencode([("path", snapshot.rel_path)])


def _get_archive_root(request: Request) -> Path:
    root = getattr(request.app.state, "archive_root", None)

    if root is None:
        orchestrator = getattr(request.app.state, "orchestrator", None)
        root = getattr(orchestrator, "archive_root", None)

    if root is None:
        root = "archive_logs"

    return Path(root).resolve()


def _list_snapshots(
    request: Request,
    camera_id: str | None = None,
    limit: int = 80,
) -> list[SnapshotInfo]:
    archive_root = _get_archive_root(request)
    snapshots_root = archive_root / "snapshots"

    if not snapshots_root.exists():
        return []

    if camera_id:
        search_root = snapshots_root / camera_id

        if not search_root.exists():
            return []
    else:
        search_root = snapshots_root

    files = [
        path
        for path in search_root.rglob("*.jpg")
        if path.is_file()
    ]

    files.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    result: list[SnapshotInfo] = []

    for path in files[:limit]:
        info = _parse_snapshot_file(
            archive_root=archive_root,
            snapshots_root=snapshots_root,
            path=path,
        )

        if info is not None:
            result.append(info)

    return result


def _get_latest_snapshot(
    request: Request,
    camera_id: str,
) -> SnapshotInfo | None:
    snapshots = _list_snapshots(
        request=request,
        camera_id=camera_id,
        limit=1,
    )

    if not snapshots:
        return None

    return snapshots[0]


def _parse_snapshot_file(
    archive_root: Path,
    snapshots_root: Path,
    path: Path,
) -> SnapshotInfo | None:
    try:
        rel_to_snapshots = path.relative_to(snapshots_root)
        rel_to_archive = path.relative_to(archive_root)
    except ValueError:
        return None

    parts = rel_to_snapshots.parts

    if len(parts) < 3:
        return None

    camera_id = parts[0]
    session_id = parts[1]

    stat = path.stat()
    saved_at = datetime.fromtimestamp(stat.st_mtime)
    saved_at_text = saved_at.strftime("%Y-%m-%d %H:%M:%S")

    filename = path.name
    stem = path.stem
    tokens = stem.split("_")

    detection_type = "detection"
    frame_id = None

    if len(tokens) >= 2 and tokens[-1] == "detection":
        detection_type = f"{tokens[-2]}_detection"

        if len(tokens) >= 3 and tokens[-3].isdigit():
            frame_id = tokens[-3]

    return SnapshotInfo(
        camera_id=camera_id,
        path=path,
        rel_path=rel_to_archive.as_posix(),
        filename=filename,
        session_id=session_id,
        saved_at=saved_at,
        saved_at_text=saved_at_text,
        detection_type=detection_type,
        frame_id=frame_id,
    )


def _render_detection_badges(
    settings,
    camera_id: str,
) -> str:
    specs = _get_detection_configs(settings, camera_id)

    if not specs:
        return '<span class="badge muted">no detections</span>'

    badges = []

    for spec in specs:
        detection_type = _value_to_str(getattr(spec, "detection_type", spec))
        ai_fps = _format_fps(_get_ai_fps(spec))

        badges.append(
            f'<span class="badge blue">'
            f'{_escape(detection_type)} <small>{_escape(ai_fps)} fps</small>'
            f'</span>'
        )

    return "\n".join(badges)


def _render_rule_badges(config: CameraConfiguration) -> str:
    rules = getattr(config.camera, "rules", []) or []

    if not rules:
        return '<span class="badge muted">no rules</span>'

    badges = []

    for rule in rules:
        if not isinstance(rule, dict):
            continue

        rule_type = rule.get("type", "rule")
        rule_name = rule.get("name", rule_type)

        if rule_type == "tripwire":
            css = "red"
            label = f"{rule_name}: {rule.get('trigger_event', '')}"
        elif rule_type == "multimodal_gate_open":
            css = "green"
            label = f"{rule_name}: {rule.get('command_event', '')}"
        else:
            css = "muted"
            label = f"{rule_name}: {rule_type}"

        badges.append(
            f'<span class="badge {css}">{_escape(label)}</span>'
        )

    return "\n".join(badges) if badges else '<span class="badge muted">no rules</span>'


def _render_camera_flags(config: CameraConfiguration) -> str:
    flags = []

    if getattr(config.camera, "audio_enabled", False):
        flags.append('<span class="pill good">audio</span>')
    else:
        flags.append('<span class="pill muted">no audio</span>')

    if getattr(config.camera, "scene_tracking", False):
        flags.append('<span class="pill warn">scene tracking</span>')
    else:
        flags.append('<span class="pill muted">static scene</span>')

    if getattr(config, "archive", False):
        flags.append('<span class="pill good">archive</span>')
    else:
        flags.append('<span class="pill muted">no archive</span>')

    return "\n".join(flags)


def _render_detection_toggle(
    detection_type: str,
    active: bool,
) -> str:
    checked = "checked" if active else ""

    return f"""
    <label class="toggle">
        <input
            type="checkbox"
            onchange="toggleDetection('{_escape(detection_type)}', this.checked)"
            {checked}
        >
        <span>{_escape(detection_type)}</span>
    </label>
    """


def _render_latest_snapshot_box(snapshot: SnapshotInfo | None) -> str:
    if snapshot is None:
        return """
        <div class="archive-mini empty">
            <div class="archive-mini-text">
                <strong>No snapshots</strong>
                <span>Archive empty</span>
            </div>
        </div>
        """

    image_url = _archive_image_url(snapshot)

    return f"""
    <a class="archive-mini" href="/archive/{_escape(snapshot.camera_id)}">
        <img src="{image_url}" alt="{_escape(snapshot.filename)}">
        <div class="archive-mini-text">
            <strong>{_escape(snapshot.detection_type)}</strong>
            <span>{_escape(snapshot.saved_at_text)}</span>
        </div>
    </a>
    """


def _base_css() -> str:
    return """
    :root {
        --bg: #080b12;
        --panel: #111827;
        --panel2: #0f172a;
        --border: rgba(148, 163, 184, 0.22);
        --text: #e5e7eb;
        --muted: #94a3b8;
        --accent: #f59e0b;
        --green: #22c55e;
        --red: #ef4444;
        --blue: #3b82f6;
        --shadow: 0 18px 55px rgba(0, 0, 0, 0.34);
    }

    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        min-height: 100vh;
        background:
            radial-gradient(circle at 15% 0%, rgba(245, 158, 11, 0.12), transparent 28%),
            radial-gradient(circle at 85% 8%, rgba(56, 189, 248, 0.10), transparent 24%),
            var(--bg);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    a {
        color: inherit;
    }

    .page {
        width: min(1800px, calc(100% - 32px));
        margin: 0 auto;
        padding: 26px 0 44px;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 20px;
        margin-bottom: 22px;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    h1 {
        margin: 0;
        font-size: clamp(30px, 4vw, 52px);
        line-height: 1;
        letter-spacing: -0.04em;
    }

    .subtitle {
        margin-top: 9px;
        color: var(--muted);
        font-size: 15px;
    }

    .nav-actions {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    .button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 38px;
        padding: 0 14px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(15, 23, 42, 0.88);
        color: var(--text);
        text-decoration: none;
        font-size: 13px;
        font-weight: 900;
    }

    .button.primary {
        background: linear-gradient(135deg, #f59e0b, #f97316);
        color: #111827;
        border-color: transparent;
    }

    .button.active {
        background: rgba(245, 158, 11, 0.18);
        border-color: rgba(245, 158, 11, 0.55);
        color: #fde68a;
    }

    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
        gap: 20px;
    }

    .camera-card,
    .side-panel,
    .archive-card {
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.94), rgba(15, 23, 42, 0.94));
        box-shadow: var(--shadow);
        overflow: hidden;
    }

    .thumb-link {
        display: block;
        position: relative;
        background: #020617;
        aspect-ratio: 16 / 9;
        overflow: hidden;
    }

    .thumb {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: contain;
        background: #020617;
    }

    .thumb-overlay {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        opacity: 0;
        background: rgba(2, 6, 23, 0.18);
        transition: 160ms ease;
    }

    .thumb-overlay span {
        padding: 10px 16px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.25);
        background: rgba(15, 23, 42, 0.78);
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.12em;
    }

    .thumb-link:hover .thumb-overlay {
        opacity: 1;
    }

    .card-body {
        padding: 16px;
    }

    .card-title-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 14px;
    }

    .card-title-row h2 {
        margin: 0;
        font-size: 21px;
        letter-spacing: -0.02em;
    }

    .camera-id {
        margin-top: 4px;
        color: var(--muted);
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 13px;
    }

    .status-dot {
        width: 12px;
        height: 12px;
        margin-top: 6px;
        border-radius: 999px;
        background: var(--green);
        box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.16);
        flex: 0 0 auto;
    }

    .meta-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 14px;
    }

    .meta-grid > div {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 10px 12px;
        background: rgba(2, 6, 23, 0.25);
    }

    .meta-label {
        display: block;
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 900;
        margin-bottom: 4px;
    }

    .section-label {
        margin: 14px 0 8px;
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-weight: 900;
    }

    .badge-row,
    .flag-row,
    .toggle-list {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .badge,
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        padding: 6px 9px;
        font-size: 12px;
        font-weight: 800;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .badge small {
        color: rgba(255,255,255,0.72);
        font-size: 10px;
        font-weight: 900;
        text-transform: uppercase;
    }

    .blue {
        background: rgba(59, 130, 246, 0.18);
        color: #bfdbfe;
    }

    .red {
        background: rgba(239, 68, 68, 0.18);
        color: #fecaca;
    }

    .green,
    .good {
        background: rgba(34, 197, 94, 0.16);
        color: #bbf7d0;
    }

    .warn {
        background: rgba(245, 158, 11, 0.16);
        color: #fde68a;
    }

    .muted {
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
    }

    .actions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 16px;
    }

    .archive-mini {
        display: grid;
        grid-template-columns: 100px 1fr;
        gap: 12px;
        align-items: center;
        text-decoration: none;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 8px;
        background: rgba(2, 6, 23, 0.25);
        min-height: 74px;
    }

    .archive-mini img {
        width: 100px;
        height: 56px;
        object-fit: cover;
        border-radius: 10px;
        background: #020617;
    }

    .archive-mini.empty {
        grid-template-columns: 1fr;
    }

    .archive-mini-text strong {
        display: block;
        font-size: 13px;
    }

    .archive-mini-text span {
        display: block;
        margin-top: 4px;
        color: var(--muted);
        font-size: 12px;
    }

    .camera-layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 380px;
        gap: 18px;
        align-items: start;
    }

    .stream-panel {
        overflow: hidden;
        border-radius: 22px;
        border: 1px solid var(--border);
        background: #020617;
        box-shadow: var(--shadow);
    }

    .stream {
        display: block;
        width: 100%;
        height: auto;
        background: #020617;
    }

    .side-panel {
        padding: 16px;
        position: sticky;
        top: 16px;
    }

    .toggle {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 8px 10px;
        border-radius: 999px;
        background: rgba(2, 6, 23, 0.36);
        border: 1px solid var(--border);
        color: #dbeafe;
        font-size: 12px;
        font-weight: 800;
        cursor: pointer;
        user-select: none;
    }

    .toggle input {
        accent-color: #f59e0b;
    }

    .archive-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 16px;
    }

    .archive-card {
        display: block;
        text-decoration: none;
    }

    .archive-card img {
        display: block;
        width: 100%;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        background: #020617;
    }

    .archive-card-body {
        padding: 12px;
    }

    .archive-card-body strong {
        display: block;
        font-size: 14px;
    }

    .archive-card-body span {
        display: block;
        margin-top: 4px;
        color: var(--muted);
        font-size: 12px;
    }

    .empty-state {
        border: 1px dashed var(--border);
        border-radius: 22px;
        padding: 44px;
        text-align: center;
        background: rgba(15, 23, 42, 0.50);
    }

    @media (max-width: 1080px) {
        .camera-layout {
            grid-template-columns: 1fr;
        }

        .side-panel {
            position: static;
        }
    }

    @media (max-width: 720px) {
        .topbar {
            align-items: flex-start;
            flex-direction: column;
        }

        .nav-actions {
            justify-content: flex-start;
        }

        .grid {
            grid-template-columns: 1fr;
        }
    }

    .archive-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    padding: 7px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(2, 6, 23, 0.34);
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
}

.archive-status::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: var(--green);
    box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.12);
}

.archive-status.checking::before {
    background: var(--accent);
    box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.14);
}

.archive-status.error::before {
    background: var(--red);
    box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.14);
}

.recent-snapshots {
    display: grid;
    gap: 8px;
}

.recent-snapshots.flash {
    animation: snapshotFlash 900ms ease-out;
}

@keyframes snapshotFlash {
    0% {
        filter: brightness(1.5);
        transform: scale(1.01);
    }

    100% {
        filter: brightness(1);
        transform: scale(1);
    }
}

.snapshot-toast {
    position: fixed;
    right: 22px;
    bottom: 22px;
    z-index: 9999;
    transform: translateY(16px);
    opacity: 0;
    pointer-events: none;
    padding: 12px 16px;
    border-radius: 14px;
    border: 1px solid rgba(34, 197, 94, 0.35);
    background: rgba(15, 23, 42, 0.94);
    color: #bbf7d0;
    font-size: 13px;
    font-weight: 900;
    box-shadow: var(--shadow);
    transition: 180ms ease;
}

.snapshot-toast.visible {
    transform: translateY(0);
    opacity: 1;
}
    """


@router.get(
    "/stream",
    summary="MJPEG stream from raw or AI camera topic",
)
async def get_live_stream(
    request: Request,
    cam: str = Query(..., description="Camera ID from registry."),
    detect: Optional[List[str]] = Query(None),
    quality: int = Query(75, ge=25, le=95),
    fps: float = Query(12.0, ge=0.5, le=30.0),
    width: Optional[int] = Query(None, ge=160, le=3840),
):
    settings = get_settings(request)
    config = _get_camera_config_or_404(settings, cam)
    _require_active_camera(config)

    allowed_detections = _get_allowed_detection_types(settings, cam)
    active_detections = _resolve_requested_detections(
        requested=detect,
        allowed=allowed_detections,
        default_all=False,
    )

    frame_bus = request.app.state.frame_bus
    camera_id = config.camera.id

    topic = f"{camera_id}_ai" if active_detections else camera_id

    handler = LiveStreamHandler(
        topic=topic,
        frame_bus=frame_bus,
        detector_provider=get_detector_provider(request),
        visible_types=active_detections,
        camera_config=config,
        audio_overlay_state=get_audio_overlay_state(request),
        jpeg_quality=quality,
        max_fps=fps,
        max_width=width,
    )

    return StreamingResponse(
        handler.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get(
    "/dashboard",
    summary="Camera dashboard",
    response_class=HTMLResponse,
)
def dashboard_page(
    request: Request,
    ai_preview: bool = Query(
        False,
        description="Use AI topics for thumbnails. Default false for lower latency.",
    ),
    settings=Depends(get_settings),
):
    active_items: list[tuple[str, CameraConfiguration]] = [
        (camera_id, config)
        for camera_id, config in settings.registry.items()
        if getattr(config.camera, "active", False)
    ]

    cards_html = []

    for camera_id, config in active_items:
        allowed_detections = _get_allowed_detection_types(settings, camera_id)
        thumbnail_detections = allowed_detections if ai_preview else []

        thumbnail_url = _build_stream_url(
            camera_id=camera_id,
            detections=thumbnail_detections,
            quality=70,
            fps=2.0,
            width=720,
        )

        ai_view_url = _build_view_url(
            camera_id=camera_id,
            detections=allowed_detections,
            raw=False,
        )

        raw_view_url = _build_view_url(
            camera_id=camera_id,
            raw=True,
        )

        latest_snapshot = _get_latest_snapshot(
            request=request,
            camera_id=camera_id,
        )

        name = _escape(getattr(config.camera, "name", camera_id))
        manufacturer = _escape(getattr(config.camera, "manufacturer", ""))
        resolution = _escape(getattr(config.camera, "resolution", ""))

        cards_html.append(
            f"""
            <article class="camera-card">
                <a class="thumb-link" href="{ai_view_url}">
                    <img class="thumb" src="{thumbnail_url}" alt="{name}">
                    <div class="thumb-overlay">
                        <span>OPEN CAMERA</span>
                    </div>
                </a>

                <div class="card-body">
                    <div class="card-title-row">
                        <div>
                            <h2>{name}</h2>
                            <div class="camera-id">{_escape(camera_id)}</div>
                        </div>
                        <span class="status-dot"></span>
                    </div>

                    <div class="meta-grid">
                        <div>
                            <span class="meta-label">Manufacturer</span>
                            <strong>{manufacturer}</strong>
                        </div>
                        <div>
                            <span class="meta-label">Resolution</span>
                            <strong>{resolution}</strong>
                        </div>
                    </div>

                    <div class="section-label">Status</div>
                    <div class="flag-row">
                        {_render_camera_flags(config)}
                    </div>

                    <div class="section-label">Detections</div>
                    <div class="badge-row">
                        {_render_detection_badges(settings, camera_id)}
                    </div>

                    <div class="section-label">Rules</div>
                    <div class="badge-row">
                        {_render_rule_badges(config)}
                    </div>

                    <div class="section-label">Latest archive snapshot</div>
                    {_render_latest_snapshot_box(latest_snapshot)}

                    <div class="actions">
                        <a class="button primary" href="{ai_view_url}">AI view</a>
                        <a class="button" href="{raw_view_url}">Raw view</a>
                        <a class="button" href="/archive/{_escape(camera_id)}">Archive</a>
                    </div>
                </div>
            </article>
            """
        )

    if cards_html:
        cards = "\n".join(cards_html)
    else:
        cards = """
        <div class="empty-state">
            <h2>No active cameras</h2>
            <p>Enable at least one camera in the registry.</p>
        </div>
        """

    preview_toggle_url = "/dashboard?ai_preview=false" if ai_preview else "/dashboard?ai_preview=true"
    preview_toggle_label = "RAW thumbnails" if ai_preview else "AI thumbnails"
    preview_badge = "AI preview" if ai_preview else "RAW preview"

    html_content = f"""
    <!doctype html>
    <html lang="pl">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Vision Core Dashboard</title>
            <style>
                {_base_css()}
            </style>
        </head>

        <body>
            <main class="page">
                <header class="topbar">
                    <div>
                        <div class="eyebrow">Vision Core · {preview_badge}</div>
                        <h1>Camera Dashboard</h1>
                        <div class="subtitle">
                            Aktywne kamery, skonfigurowane detekcje, reguły i ostatnie snapshoty.
                        </div>
                    </div>

                    <div class="nav-actions">
                        <a class="button" href="{preview_toggle_url}">{preview_toggle_label}</a>
                        <a class="button" href="/archive">Archive</a>
                    </div>
                </header>

                <section class="grid">
                    {cards}
                </section>
            </main>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get(
    "/view",
    summary="HTML camera preview page",
    response_class=HTMLResponse,
)
def view_camera_page(
    request: Request,
    cam: str = Query(..., examples=["dom_c1"]),
    detect: Optional[List[str]] = Query(None),
    raw: bool = Query(False),
    settings=Depends(get_settings),
):
    config = _get_camera_config_or_404(settings, cam)
    _require_active_camera(config)

    camera_id = config.camera.id
    allowed_detections = _get_allowed_detection_types(settings, camera_id)

    if raw:
        active_detections: list[str] = []
    else:
        active_detections = _resolve_requested_detections(
            requested=detect,
            allowed=allowed_detections,
            default_all=True,
        )

    stream_url = _build_stream_url(
        camera_id=camera_id,
        detections=active_detections,
        quality=78,
        fps=12.0,
        width=None,
    )

    ai_url = _build_view_url(
        camera_id=camera_id,
        detections=allowed_detections,
        raw=False,
    )

    raw_url = _build_view_url(
        camera_id=camera_id,
        raw=True,
    )

    archive_url = f"/archive/{camera_id}"

    name = _escape(getattr(config.camera, "name", camera_id))
    manufacturer = _escape(getattr(config.camera, "manufacturer", ""))
    resolution = _escape(getattr(config.camera, "resolution", ""))

    ai_button_class = "button primary active" if not raw else "button"
    raw_button_class = "button primary active" if raw else "button"

    detection_toggles = "\n".join(
        _render_detection_toggle(
            detection_type=detection_type,
            active=detection_type in active_detections,
        )
        for detection_type in allowed_detections
    )

    if not detection_toggles:
        detection_toggles = '<span class="badge muted">no configured detections</span>'

    recent_snapshots = _list_snapshots(
        request=request,
        camera_id=camera_id,
        limit=4,
    )

    recent_snapshot_items = []

    for snapshot in recent_snapshots:
        image_url = _archive_image_url(snapshot)

        recent_snapshot_items.append(
            f"""
            <a class="archive-mini" href="{image_url}" target="_blank">
                <img src="{image_url}" alt="{_escape(snapshot.filename)}">
                <div class="archive-mini-text">
                    <strong>{_escape(snapshot.detection_type)}</strong>
                    <span>{_escape(snapshot.saved_at_text)}</span>
                </div>
            </a>
            """
        )

    recent_snapshots_html = "\n".join(recent_snapshot_items) if recent_snapshot_items else """
    <div class="archive-mini empty">
        <div class="archive-mini-text">
            <strong>No snapshots</strong>
            <span>No archive files for this camera yet.</span>
        </div>
    </div>
    """

    active_detections_json = json.dumps(active_detections)
    camera_id_json = json.dumps(camera_id)

    html_content = f"""
    <!doctype html>
    <html lang="pl">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{name} | Vision Core</title>
            <style>
                {_base_css()}
            </style>
        </head>

        <body>
            <main class="page">
                <header class="topbar">
                    <div>
                        <div class="eyebrow">Vision Core · Camera View</div>
                        <h1>{name}</h1>
                        <div class="subtitle">
                            {_escape(camera_id)} · {manufacturer} · {resolution}
                        </div>
                    </div>

                    <div class="nav-actions">
                        <a class="button" href="/dashboard">Dashboard</a>
                        <a class="button" href="{archive_url}">Archive</a>
                    </div>
                </header>

                <section class="camera-layout">
                    <div class="stream-panel">
                        <img class="stream" src="{stream_url}">
                    </div>

                    <aside class="side-panel">
                        <div class="actions">
                            <a class="{ai_button_class}" href="{ai_url}">AI view</a>
                            <a class="{raw_button_class}" href="{raw_url}">Raw view</a>
                        </div>

                        <div class="section-label">Status</div>
                        <div class="flag-row">
                            {_render_camera_flags(config)}
                        </div>

                        <div class="section-label">Aktywne detekcje</div>
                        <div class="toggle-list">
                            {detection_toggles}
                        </div>

                        <div class="section-label">Detectors config</div>
                        <div class="badge-row">
                            {_render_detection_badges(settings, camera_id)}
                        </div>

                        <div class="section-label">Rules</div>
                        <div class="badge-row">
                            {_render_rule_badges(config)}
                        </div>

                        <div class="section-label">Recent snapshots</div>

                        <div id="archiveStatus" class="archive-status">
                            Watching archive...
                        </div>

                        <div
                            id="recentSnapshots"
                            class="recent-snapshots"
                            data-camera-id="{_escape(camera_id)}"
                        >
                            {recent_snapshots_html}
                        </div>

                        <div id="snapshotToast" class="snapshot-toast">
                            Snapshot saved
                        </div>
                        </div>
                    </aside>
                </section>
            </main>

            <script>
                const cameraId = {camera_id_json};
                const activeDetections = new Set({active_detections_json});

                function toggleDetection(detectionType, enabled) {{
                    const params = new URLSearchParams(window.location.search);

                    params.set("cam", cameraId);
                    params.set("raw", "false");

                    const values = new Set(activeDetections);

                    if (enabled) {{
                        values.add(detectionType);
                    }} else {{
                        values.delete(detectionType);
                    }}

                    params.delete("detect");

                    if (values.size === 0) {{
                        params.append("detect", "none");
                    }} else {{
                        for (const value of values) {{
                            params.append("detect", value);
                        }}
                    }}

                    window.location.search = params.toString();
                }}
            </script>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get(
    "/archive",
    summary="Archive snapshots overview",
    response_class=HTMLResponse,
)
def archive_page(
    request: Request,
    limit: int = Query(80, ge=1, le=300),
    settings=Depends(get_settings),
):
    snapshots = _list_snapshots(
        request=request,
        camera_id=None,
        limit=limit,
    )

    cards = _render_snapshot_cards(snapshots)

    if not cards:
        cards = """
        <div class="empty-state">
            <h2>No snapshots</h2>
            <p>Nothing has been saved in archive_logs/snapshots yet.</p>
        </div>
        """

    camera_buttons = []

    for camera_id, config in settings.registry.items():
        camera_buttons.append(
            f'<a class="button" href="/archive/{_escape(camera_id)}">{_escape(camera_id)}</a>'
        )

    html_content = f"""
    <!doctype html>
    <html lang="pl">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Archive | Vision Core</title>
            <style>
                {_base_css()}
            </style>
        </head>

        <body>
            <main class="page">
                <header class="topbar">
                    <div>
                        <div class="eyebrow">Vision Core · Archive</div>
                        <h1>Snapshots</h1>
                        <div class="subtitle">
                            Ostatnie zapisane detekcje z archive_logs/snapshots.
                        </div>
                    </div>

                    <div class="nav-actions">
                        <a class="button" href="/dashboard">Dashboard</a>
                        {"".join(camera_buttons)}
                    </div>
                </header>

                <section class="archive-grid">
                    {cards}
                </section>
            </main>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get(
    "/archive/{camera_id}",
    summary="Archive snapshots for camera",
    response_class=HTMLResponse,
)
def archive_camera_page(
    request: Request,
    camera_id: str,
    limit: int = Query(120, ge=1, le=500),
    settings=Depends(get_settings),
):
    config = settings.registry.get(camera_id)

    camera_name = camera_id

    if config is not None:
        camera_name = getattr(config.camera, "name", camera_id)

    snapshots = _list_snapshots(
        request=request,
        camera_id=camera_id,
        limit=limit,
    )

    cards = _render_snapshot_cards(snapshots)

    if not cards:
        cards = f"""
        <div class="empty-state">
            <h2>No snapshots for {_escape(camera_id)}</h2>
            <p>No files found in archive_logs/snapshots/{_escape(camera_id)}.</p>
        </div>
        """

    view_url = _build_view_url(
        camera_id=camera_id,
        raw=False,
        detections=_get_allowed_detection_types(settings, camera_id),
    )

    html_content = f"""
    <!doctype html>
    <html lang="pl">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Archive {camera_id} | Vision Core</title>
            <style>
                {_base_css()}
            </style>
        </head>

        <body>
            <main class="page">
                <header class="topbar">
                    <div>
                        <div class="eyebrow">Vision Core · Archive</div>
                        <h1>{_escape(camera_name)}</h1>
                        <div class="subtitle">
                            Snapshoty dla kamery {_escape(camera_id)}.
                        </div>
                    </div>

                    <div class="nav-actions">
                        <a class="button" href="/dashboard">Dashboard</a>
                        <a class="button" href="/archive">All archive</a>
                        <a class="button primary" href="{view_url}">Camera view</a>
                    </div>
                </header>

                <section class="archive-grid">
                    {cards}
                </section>
            </main>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)


def _render_snapshot_cards(snapshots: list[SnapshotInfo]) -> str:
    cards = []

    for snapshot in snapshots:
        image_url = _archive_image_url(snapshot)

        frame_text = f"frame {snapshot.frame_id}" if snapshot.frame_id else "frame -"

        cards.append(
            f"""
            <a class="archive-card" href="{image_url}" target="_blank">
                <img src="{image_url}" alt="{_escape(snapshot.filename)}">
                <div class="archive-card-body">
                    <strong>{_escape(snapshot.camera_id)} · {_escape(snapshot.detection_type)}</strong>
                    <span>{_escape(snapshot.saved_at_text)}</span>
                    <span>{_escape(frame_text)} · {_escape(snapshot.session_id)}</span>
                </div>
            </a>
            """
        )

    return "\n".join(cards)


@router.get(
    "/archive-image",
    summary="Read snapshot image from archive_logs",
)
def archive_image(
    request: Request,
    path: str = Query(...),
):
    archive_root = _get_archive_root(request)
    requested = (archive_root / path).resolve()

    try:
        requested.relative_to(archive_root)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Path outside archive root is not allowed.",
        )

    if not requested.exists() or not requested.is_file():
        raise HTTPException(
            status_code=404,
            detail="Snapshot image not found.",
        )

    return FileResponse(requested)




@router.get(
    "/archive/recent.json",
    summary="Recent archive snapshots as JSON",
)
def archive_recent_json(
    request: Request,
    camera_id: str | None = Query(None),
    limit: int = Query(8, ge=1, le=100),
):
    snapshots = _list_snapshots(
        request=request,
        camera_id=camera_id,
        limit=limit,
    )

    payload = {
        "status": "ok",
        "camera_id": camera_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "snapshots": [
            _snapshot_to_dict(snapshot)
            for snapshot in snapshots
        ],
    }

    return JSONResponse(
        content=payload,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )