"""Visualisation API — load .mat, stream rendered frames, export screenshots."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import numpy as np
from pydantic import BaseModel

router = APIRouter(prefix="/api/viz", tags=["visualization"])
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Lazy-initialised global session (one per server process).
_session = None


def _get_session():
    global _session
    if _session is None:
        from jaxmech.modules.visualization.session import VizSession
        _session = VizSession()
    return _session


def _resolve(raw_path: str) -> Path:
    """Resolve and validate a file path within the project tree."""
    text = str(raw_path).strip()
    if text.startswith("/mnt/") and len(text) > 7:
        drive = text[5].upper()
        tail = text[7:].replace("/", "\\")
        text = f"{drive}:\\{tail}"
    path = Path(text).resolve()
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise HTTPException(403, f"Access denied: {raw_path}")
    if not path.exists():
        raise HTTPException(404, f"File not found: {raw_path}")
    return path


# ── Pydantic request models ──────────────────────────────────────────

class LoadRequest(BaseModel):
    mat_path: str
    slot: str = "a"
    selected_fields: Optional[list[str]] = None


class InspectRequest(BaseModel):
    mat_path: str


class SetFieldRequest(BaseModel):
    key: str
    component: Optional[int] = None
    frame: Optional[int] = None
    slot: str = "a"


class SetFrameRequest(BaseModel):
    frame: int
    slot: str = "a"


class SetLegendRequest(BaseModel):
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    cmap: Optional[str] = None
    slot: str = "a"


class SetViewRequest(BaseModel):
    preset: Optional[str] = None
    camera: Optional[dict] = None
    slot: str = "a"


class ScreenshotRequest(BaseModel):
    scale: int = 4
    dpi: int = 300
    transparent: bool = False
    slot: str = "a"


class CompareRequest(BaseModel):
    enabled: bool


class SyncCameraRequest(BaseModel):
    enabled: bool


class ShowEdgesRequest(BaseModel):
    show: bool
    slot: str = "a"


class DeformRequest(BaseModel):
    scale: float = 0.0
    slot: str = "a"


class BoxplotRequest(BaseModel):
    absolute: bool = False


class DiffRequest(BaseModel):
    absolute: bool = False
    output_mat: bool = False


# ── REST endpoints ───────────────────────────────────────────────────

@router.post("/load")
async def load_mat(req: LoadRequest):
    """Load a .mat file into a scene slot."""
    path = _resolve(req.mat_path)
    session = _get_session()
    try:
        if req.slot != "b":
            from jaxmech.modules.visualization.mat_reader import is_validation_mat
            if is_validation_mat(path):
                pair = session.load_validation_pair(str(path), selected_fields=req.selected_fields)
                await _close_slot_ws("a", reason="Scene reloaded")
                await _close_slot_ws("b", reason="Scene reloaded")
                scene_a = pair.get("scene_a") or {}
                return {"ok": True, **scene_a, **pair}
        if req.slot == "b":
            info = session.load_b(str(path), selected_fields=req.selected_fields)
        else:
            info = session.load_a(str(path), selected_fields=req.selected_fields)
        await _close_slot_ws(req.slot, reason="Scene reloaded")
    except Exception as exc:
        raise HTTPException(400, f"Failed to load: {exc}")
    return {"ok": True, **info}


@router.post("/inspect")
async def inspect_mat(req: InspectRequest):
    """Inspect a .mat file and return candidate field variables."""
    path = _resolve(req.mat_path)
    try:
        from jaxmech.modules.visualization.mat_reader import inspect_mat_fields
        catalog = inspect_mat_fields(path)
    except Exception as exc:
        raise HTTPException(400, f"Failed to inspect MAT: {exc}")
    return {"ok": True, **catalog}


@router.post("/set-field")
async def set_field(req: SetFieldRequest):
    """Switch the active field variable."""
    scene = _get_session().get_scene(req.slot)
    if scene is None:
        raise HTTPException(400, "No scene loaded in this slot.")
    scene.set_field(req.key, component=req.component, frame=req.frame)
    return {"ok": True, **scene.info}


@router.post("/set-frame")
async def set_frame(req: SetFrameRequest):
    scene = _get_session().get_scene(req.slot)
    if scene is None:
        raise HTTPException(400, "No scene loaded.")
    scene.set_frame(req.frame)
    return {"ok": True, **scene.info}


@router.post("/set-legend")
async def set_legend(req: SetLegendRequest):
    scene = _get_session().get_scene(req.slot)
    if scene is None:
        raise HTTPException(400, "No scene loaded.")
    if req.vmin is not None or req.vmax is not None:
        scene.set_clim(req.vmin, req.vmax)
    if req.cmap is not None:
        scene.set_colormap(req.cmap)
    return {"ok": True, **scene.info}


@router.post("/set-view")
async def set_view(req: SetViewRequest):
    scene = _get_session().get_scene(req.slot)
    if scene is None:
        raise HTTPException(400, "No scene loaded.")
    if req.preset:
        scene.set_preset_view(req.preset)
    elif req.camera:
        scene.set_camera_state(req.camera)
    return {"ok": True}


@router.post("/show-edges")
async def show_edges(req: ShowEdgesRequest):
    scene = _get_session().get_scene(req.slot)
    if scene is None:
        raise HTTPException(400, "No scene loaded.")
    scene.set_show_edges(req.show)
    return {"ok": True}


@router.post("/deform")
async def deform(req: DeformRequest):
    """Set the displacement magnification factor."""
    scene = _get_session().get_scene(req.slot)
    if scene is None:
        raise HTTPException(400, "No scene loaded.")
    scene.set_deform_scale(req.scale)
    return {"ok": True, **scene.info}


@router.post("/screenshot")
async def screenshot(req: ScreenshotRequest):
    """Return a high-resolution PNG screenshot."""
    scene = _get_session().get_scene(req.slot)
    if scene is None:
        raise HTTPException(400, "No scene loaded.")
    from jaxmech.modules.visualization.screenshot import screenshot_bytes
    png = screenshot_bytes(
        scene, scale=req.scale, dpi=req.dpi,
        transparent_background=req.transparent,
    )
    return Response(content=png, media_type="image/png", headers={
        "Content-Disposition": "attachment; filename=screenshot.png",
    })


@router.post("/compare")
async def set_compare(req: CompareRequest):
    session = _get_session()
    if session.compare_locked and not req.enabled:
        return {
            "ok": False,
            "compare_locked": True,
            "detail": "Validation MAT uses locked embedded ODB A/B comparison.",
            **session.info,
        }
    if not req.enabled:
        session.close_b()
    session.set_compare_mode(req.enabled)
    return {"ok": True, **session.info}


@router.post("/sync-camera")
async def sync_camera(req: SyncCameraRequest):
    session = _get_session()
    session.set_sync_camera(req.enabled)
    if req.enabled and session.compare_mode:
        await _push_peer_frame("a")
    return {"ok": True, "sync_camera": session.sync_camera}


@router.get("/info")
async def info():
    """Return current session state."""
    return _get_session().info


@router.get("/render/{slot}")
async def render_frame(slot: str = "a"):
    """Return a single rendered JPEG frame."""
    scene = _get_session().get_scene(slot)
    if scene is None:
        raise HTTPException(400, "No scene loaded.")
    jpg = scene.render_jpeg()
    return Response(content=jpg, media_type="image/jpeg")


def _boxplot_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise HTTPException(400, "No finite scalar values available for boxplot.")

    q1, median, q3 = np.percentile(arr, [25.0, 50.0, 75.0])
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    inlier = arr[(arr >= low_fence) & (arr <= high_fence)]
    whisker_low = float(inlier.min()) if inlier.size else float(arr.min())
    whisker_high = float(inlier.max()) if inlier.size else float(arr.max())
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(arr.max()),
        "iqr": float(iqr),
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "outlier_count": int(arr.size - inlier.size),
    }


def _selection_payload(scene) -> dict:
    info = scene.info
    return {
        "source": info.get("source", ""),
        "field": info.get("active_field", ""),
        "field_label": info.get("active_field_label", ""),
        "scalar_label": info.get("active_scalar_label", ""),
        "component": info.get("active_component"),
        "frame": info.get("active_frame", 0),
    }


@router.post("/boxplot")
async def compare_boxplot(req: BoxplotRequest):
    session = _get_session()
    scene_a = session.get_scene("a")
    scene_b = session.get_scene("b")
    if scene_a is None or scene_b is None:
        raise HTTPException(400, "Compare mode requires both scene A and scene B to be loaded.")

    values_a = scene_a.get_scalar_data()
    values_b = scene_b.get_scalar_data()
    if values_a.shape != values_b.shape:
        raise HTTPException(
            400,
            f"Current scalar fields are not shape-compatible: {values_a.shape} vs {values_b.shape}",
        )

    diff = np.asarray(values_b, dtype=np.float64) - np.asarray(values_a, dtype=np.float64)
    if req.absolute:
        diff = np.abs(diff)

    return {
        "ok": True,
        "absolute": req.absolute,
        "label": "|B - A|" if req.absolute else "B - A",
        "selection_a": _selection_payload(scene_a),
        "selection_b": _selection_payload(scene_b),
        "stats": _boxplot_stats(diff),
    }


@router.post("/diff")
async def compare_diff(req: DiffRequest):
    """Compute the displayed scalar Diff, register it as a temporary field, and optionally export MAT."""
    session = _get_session()
    try:
        from jaxmech.modules.visualization.compare import compute_session_diff
        payload = compute_session_diff(
            session,
            absolute=req.absolute,
            output_mat=req.output_mat,
        )
    except Exception as exc:
        raise HTTPException(400, f"Failed to compute Diff: {exc}")
    await _push_slot_state("a", render=True)
    await _push_slot_state("b", render=True)
    return payload


@router.post("/diff-boxplot")
async def diff_boxplot():
    """Return boxplot statistics for the currently registered Diff field."""
    try:
        from jaxmech.modules.visualization.compare import registered_diff_summary
        return registered_diff_summary(_get_session())
    except Exception as exc:
        raise HTTPException(400, f"Failed to read Diff: {exc}")


@router.post("/diff-mat")
async def diff_mat():
    """Download the currently registered Diff field as a MAT file."""
    try:
        from jaxmech.modules.visualization.compare import registered_diff_mat
        filename, data = registered_diff_mat(_get_session())
    except Exception as exc:
        raise HTTPException(400, f"Failed to save Diff: {exc}")
    return Response(content=data, media_type="application/octet-stream", headers={
        "Content-Disposition": f'attachment; filename="{filename}"',
    })


# ── WebSocket registry for cross-slot push (real-time camera sync) ───

_ws_registry: dict[str, WebSocket] = {}


async def _close_slot_ws(slot: str, *, reason: str = "Scene changed") -> None:
    ws = _ws_registry.get(slot)
    if ws is None:
        return
    try:
        await ws.close(code=4001, reason=reason)
    except Exception:
        pass
    if _ws_registry.get(slot) is ws:
        del _ws_registry[slot]


async def _push_peer_frame(slot: str) -> None:
    """Render and push a JPEG frame to the *other* slot's WebSocket."""
    peer = "b" if slot == "a" else "a"
    peer_ws = _ws_registry.get(peer)
    if peer_ws is None:
        return
    session = _get_session()
    peer_scene = session.get_scene(peer)
    if peer_scene is None:
        return
    try:
        jpg = peer_scene.render_jpeg()
        await peer_ws.send_bytes(jpg)
    except Exception:
        pass


async def _push_slot_state(slot: str, *, render: bool = True) -> None:
    ws = _ws_registry.get(slot)
    if ws is None:
        return
    scene = _get_session().get_scene(slot)
    if scene is None:
        return
    try:
        if render:
            await ws.send_bytes(scene.render_jpeg())
        await ws.send_text(json.dumps(scene.info))
    except Exception:
        pass


# ── WebSocket for interactive streaming ──────────────────────────────

@router.websocket("/ws/{slot}")
async def viz_ws(ws: WebSocket, slot: str = "a"):
    """Interactive visualisation WebSocket.

    Client sends JSON commands; server pushes rendered JPEG frames.
    Commands:
      {"action": "orbit", "dx": ..., "dy": ...}
      {"action": "pan",   "dx": ..., "dy": ...}
      {"action": "zoom",  "factor": ...}
      {"action": "set_field", "key": ..., "component": ..., "frame": ...}
      {"action": "set_frame", "frame": ...}
      {"action": "set_legend", "vmin": ..., "vmax": ...}
      {"action": "set_view", "preset": ...}
      {"action": "render"}
    """
    await ws.accept()
    session = _get_session()
    scene = session.get_scene(slot)
    if scene is None:
        await ws.close(code=4000, reason="No scene loaded in this slot.")
        return

    _ws_registry[slot] = ws
    try:
        # Send initial frame
        jpg = scene.render_jpeg()
        await ws.send_bytes(jpg)
        await ws.send_text(json.dumps(scene.info))

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = msg.get("action", "")
            send_info = False
            did_sync = False
            scene = session.get_scene(slot)
            if scene is None:
                await ws.close(code=4000, reason="No scene loaded in this slot.")
                return

            if action == "orbit":
                scene.orbit(float(msg.get("dx", 0)), float(msg.get("dy", 0)))
                if session.sync_camera and session.compare_mode:
                    session.mirror_camera(slot)
                    did_sync = True
            elif action == "pan":
                scene.pan(float(msg.get("dx", 0)), float(msg.get("dy", 0)))
                if session.sync_camera and session.compare_mode:
                    session.mirror_camera(slot)
                    did_sync = True
            elif action == "zoom":
                scene.zoom(float(msg.get("factor", 1.0)))
                if session.sync_camera and session.compare_mode:
                    session.mirror_camera(slot)
                    did_sync = True
            elif action == "set_field":
                scene.set_field(
                    msg["key"],
                    component=msg.get("component"),
                    frame=msg.get("frame"),
                )
                send_info = True
            elif action == "set_frame":
                scene.set_frame(int(msg.get("frame", 0)))
                send_info = True
            elif action == "set_legend":
                scene.set_clim(msg.get("vmin"), msg.get("vmax"))
                if "cmap" in msg:
                    scene.set_colormap(msg["cmap"])
                send_info = True
            elif action == "set_view":
                if "preset" in msg:
                    scene.set_preset_view(msg["preset"])
                elif "camera" in msg:
                    scene.set_camera_state(msg["camera"])
                if session.sync_camera and session.compare_mode:
                    session.mirror_camera(slot)
                    did_sync = True
            elif action == "show_edges":
                scene.set_show_edges(bool(msg.get("show", False)))
                send_info = True
            elif action == "show_overlay":
                scene.set_show_overlay(bool(msg.get("show", False)))
                send_info = True
            elif action == "deform":
                scene.set_deform_scale(float(msg.get("scale", 0.0)))
                send_info = True
            elif action == "animate":
                n_frames = msg.get("n_frames", 0)
                delay = float(msg.get("delay", 0.2))
                if n_frames > 0:
                    for fr in range(n_frames):
                        scene.set_frame(fr)
                        jpg = scene.render_jpeg()
                        await ws.send_bytes(jpg)
                        await ws.send_text(json.dumps(scene.info))
                        await asyncio.sleep(delay)
                    continue
            elif action == "render":
                pass  # just re-render below
            elif action == "info":
                await ws.send_text(json.dumps(scene.info))
                continue
            else:
                continue

            # Push updated frame after any mutation
            jpg = scene.render_jpeg()
            await ws.send_bytes(jpg)
            if send_info:
                await ws.send_text(json.dumps(scene.info))

            if did_sync:
                await _push_peer_frame(slot)

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
    finally:
        if _ws_registry.get(slot) is ws:
            del _ws_registry[slot]
