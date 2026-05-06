"""Analysis task API — submit and monitor computation tasks."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from jaxmech.web.services.task_runner import task_manager

router = APIRouter(tags=["tasks"])
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_local_file_path(raw_path: str) -> Path:
    path_text = str(raw_path).strip()
    if path_text.startswith("/mnt/") and len(path_text) > 7:
        drive = path_text[5].upper()
        tail = path_text[7:].replace("/", "\\")
        path_text = f"{drive}:\\{tail}"
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise HTTPException(403, f"Access denied: {raw_path}")
    return path


class RunRequest(BaseModel):
    module: str
    args: list[str] = []


class DeleteTasksRequest(BaseModel):
    ids: list[str] = []


@router.post("/api/run")
async def submit_task(body: RunRequest):
    """Submit a computation task (any module)."""
    task = task_manager.create_task(body.module, body.args)
    asyncio.create_task(task_manager.run_task(task))
    return task.to_dict()


@router.get("/api/tasks")
async def list_tasks(limit: int = 20):
    return task_manager.list_tasks(limit)


@router.post("/api/tasks/delete")
async def delete_tasks(body: DeleteTasksRequest):
    return task_manager.delete_tasks(body.ids)


@router.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = task_manager.get_task(task_id)
    if task is None:
        return {"error": "Task not found"}
    result = task.to_dict()
    result["logs"] = task.logs
    result["config"] = task_manager.task_config_snapshot(task)
    return result


@router.post("/api/tasks/{task_id}/stop")
async def stop_task(task_id: str):
    result = await task_manager.stop_task(task_id)
    if not result.get("ok"):
        raise HTTPException(404, result.get("error", "Task not found"))
    return result


@router.post("/api/tasks/{task_id}/open-folder")
async def open_task_folder(task_id: str):
    task = task_manager.get_task(task_id)
    if task is None or not task.task_dir:
        raise HTTPException(404, "Task not found")
    folder = Path(task.task_dir).resolve()
    if not folder.is_dir():
        raise HTTPException(404, f"Task folder not found: {folder}")
    if folder != task_manager.task_root.resolve() and task_manager.task_root.resolve() not in folder.parents:
        raise HTTPException(403, "Access denied")
    subprocess.Popen(["explorer.exe", str(folder)])
    return {"ok": True, "path": str(folder)}


@router.get("/api/tasks/{task_id}/logs.txt")
async def download_task_log(task_id: str):
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    log_path = task_manager.rewrite_task_log_text(task)
    return FileResponse(
        str(log_path),
        media_type="text/plain; charset=utf-8",
        filename=f"{task.id}_run.log.txt",
    )


@router.get("/api/files")
async def get_local_file(path: str):
    file_path = _resolve_local_file_path(path)
    if not file_path.is_file():
        raise HTTPException(404, f"File not found: {path}")
    return FileResponse(str(file_path))


@router.get("/api/files/exists")
async def local_file_exists(path: str):
    file_path = _resolve_local_file_path(path)
    return {"exists": file_path.is_file(), "path": str(file_path)}


@router.post("/api/files/open-folder")
async def open_local_file_folder(path: str):
    file_path = _resolve_local_file_path(path)
    if file_path.is_file():
        folder = file_path.parent
    elif file_path.is_dir():
        folder = file_path
    else:
        raise HTTPException(404, f"File not found: {path}")
    subprocess.Popen(["explorer.exe", str(folder)])
    return {"ok": True, "path": str(folder)}


@router.websocket("/ws/tasks/{task_id}/logs")
async def task_logs_ws(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time log streaming."""
    await websocket.accept()
    task = task_manager.get_task(task_id)
    if task is None:
        await websocket.send_json({"type": "error", "text": "Task not found"})
        await websocket.close()
        return

    # Do NOT replay historical logs — each page view starts with a clean console.
    # For a finished task, just send the final status so the UI can update the badge.
    if task.status.value in ("success", "failed", "stopped"):
        await websocket.send_json({
            "type": "status",
            "status": task.status.value,
            "exit_code": task.exit_code,
            "error_summary": task.error_summary,
        })
        # Then replay only the last 500 log entries so user can inspect results
        for entry in task.logs[-500:]:
            await websocket.send_json(entry)
        await websocket.close()
        return

    # Subscribe for new logs FIRST, then replay history.
    # This order guarantees no gap: anything produced between the snapshot
    # and the subscribe call will be in the queue.
    queue = task_manager.subscribe(task_id)

    # Replay logs that were already collected before the WS connected.
    for entry in list(task.logs):
        await websocket.send_json(entry)

    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("type") == "status" and msg.get("status") in ("success", "failed", "stopped"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        task_manager.unsubscribe(task_id, queue)
