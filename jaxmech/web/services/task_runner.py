"""Task runner — async subprocess manager with WebSocket log push."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from jaxmech.web.services.wsl_bridge import run_wsl_module_async


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    module: str
    args: list[str]
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    exit_code: Optional[int] = None
    logs: list[dict[str, str]] = field(default_factory=list)
    artifacts: list[dict[str, str]] = field(default_factory=list)
    error_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "module": self.module,
            "args": self.args,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "log_count": len(self.logs),
            "artifacts": self.artifacts,
            "error_summary": self.error_summary,
        }


class TaskManager:
    """Manages a pool of async subprocess tasks."""

    def __init__(self, max_concurrent: int = 2):
        self._tasks: dict[str, Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def create_task(self, module: str, args: list[str]) -> Task:
        task = Task(
            id=str(uuid.uuid4())[:8],
            module=module,
            args=args,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._tasks[task.id] = task
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks(self, limit: int = 20) -> list[dict]:
        items = sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True,
        )[:limit]
        return [t.to_dict() for t in items]

    def subscribe(self, task_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(task_id, []).append(q)
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue):
        subs = self._subscribers.get(task_id, [])
        if q in subs:
            subs.remove(q)

    async def _push(self, task_id: str, msg: dict):
        for q in self._subscribers.get(task_id, []):
            await q.put(msg)

    @staticmethod
    def _parse_artifact_line(line: str) -> dict[str, str] | None:
        prefix = "__ARTIFACT__|"
        if not line.startswith(prefix):
            return None
        payload = line[len(prefix):]
        parts = payload.split("|", 2)
        if len(parts) < 2:
            return None
        kind = parts[0].strip() or "file"
        path = parts[1].strip()
        if not path:
            return None
        label = parts[2].strip() if len(parts) >= 3 else Path(path).name
        return {"kind": kind, "path": path, "label": label}

    async def run_task(self, task: Task) -> None:
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat(timespec="seconds")
            await self._push(task.id, {"type": "status", "status": "running"})

            async def on_stdout(line: str):
                artifact = self._parse_artifact_line(line)
                if artifact is not None:
                    if not any(item.get("path") == artifact["path"] for item in task.artifacts):
                        task.artifacts.append(artifact)
                    await self._push(task.id, {"type": "artifact", "artifact": artifact})
                    info_entry = {"type": "info", "text": f"生成结果产物: {artifact['label']}"}
                    task.logs.append(info_entry)
                    await self._push(task.id, info_entry)
                    return
                entry = {"type": "stdout", "text": line}
                task.logs.append(entry)
                await self._push(task.id, entry)

            async def on_stderr(line: str):
                entry = {"type": "stderr", "text": line}
                task.logs.append(entry)
                await self._push(task.id, entry)

            try:
                exit_code = await run_wsl_module_async(
                    task.module, task.args,
                    on_stdout=on_stdout, on_stderr=on_stderr,
                )
                task.exit_code = exit_code
                task.status = TaskStatus.SUCCESS if exit_code == 0 else TaskStatus.FAILED
                if exit_code != 0:
                    # Extract last few stderr lines as error summary
                    stderr_lines = [
                        e["text"] for e in task.logs if e["type"] == "stderr"
                    ]
                    task.error_summary = "\n".join(stderr_lines[-5:])
            except Exception as exc:
                task.status = TaskStatus.FAILED
                task.error_summary = str(exc)

            task.finished_at = datetime.now().isoformat(timespec="seconds")
            await self._push(task.id, {
                "type": "status",
                "status": task.status.value,
                "exit_code": task.exit_code,
                "error_summary": task.error_summary,
            })


# Singleton
task_manager = TaskManager()
