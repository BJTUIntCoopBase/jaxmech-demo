"""Task runner — async subprocess manager with WebSocket log push."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from jaxmech.env.workflow_files import resolve_workflow_target
from jaxmech.web.services.wsl_bridge import run_wsl_module_async

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = PROJECT_ROOT / "Cache"
TASK_RUN_ROOT = CACHE_ROOT / "WebTaskRuns"
LEGACY_TASK_RUN_ROOT = PROJECT_ROOT / "WebTaskRuns"
MAX_PERSISTED_TASKS = 100


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


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
    task_dir: str = ""

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
            "task_dir": self.task_dir,
        }


class TaskManager:
    """Manages a pool of async subprocess tasks."""

    def __init__(self, max_concurrent: int = 2, task_root: Path | None = None):
        self._tasks: dict[str, Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._stop_requested: set[str] = set()
        use_default_root = task_root is None
        self.task_root = task_root or TASK_RUN_ROOT
        self.task_root.mkdir(parents=True, exist_ok=True)
        if use_default_root:
            self._migrate_legacy_task_root()
        self._load_persisted_tasks()
        self._prune_task_dirs()

    @staticmethod
    def _slug(text: str) -> str:
        tail = str(text or "task").split(".")[-1]
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", tail).strip("._")
        return slug[:40] or "task"

    def _migrate_legacy_task_root(self) -> None:
        try:
            legacy_root = LEGACY_TASK_RUN_ROOT.resolve()
            task_root = self.task_root.resolve()
        except OSError:
            return
        if legacy_root == task_root or not legacy_root.is_dir():
            return
        for child in legacy_root.iterdir():
            target = self.task_root / child.name
            if target.exists():
                stem = child.stem
                suffix = child.suffix
                index = 1
                while target.exists():
                    target = self.task_root / f"{stem}_legacy{index}{suffix}"
                    index += 1
            try:
                shutil.move(str(child), str(target))
            except OSError:
                continue
        try:
            legacy_root.rmdir()
        except OSError:
            pass

    def _task_manifest_path(self, task: Task) -> Path:
        return Path(task.task_dir) / "manifest.json"

    def _task_log_jsonl_path(self, task: Task) -> Path:
        return Path(task.task_dir) / "logs.jsonl"

    def task_log_text_path(self, task: Task) -> Path:
        return Path(task.task_dir) / "run.log"

    @staticmethod
    def _path_text_to_local(path_text: str) -> Path:
        text = str(path_text or "").strip()
        if text.startswith("/mnt/") and len(text) > 7:
            drive = text[5].upper()
            tail = text[7:].replace("/", "\\")
            text = f"{drive}:\\{tail}"
        path = Path(text)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @classmethod
    def _project_relative_text(cls, path_text: str) -> str:
        text = str(path_text or "").strip()
        if not text:
            return ""
        try:
            path = cls._path_text_to_local(text)
            rel = path.relative_to(PROJECT_ROOT.resolve())
        except Exception:
            return text
        return rel.as_posix()

    @classmethod
    def _normalize_artifact(cls, artifact: dict[str, str]) -> dict[str, str]:
        out = dict(artifact)
        if out.get("path"):
            out["path"] = cls._project_relative_text(str(out["path"]))
        return out

    @staticmethod
    def _task_status_flag(status: TaskStatus) -> str:
        return "completed" if status == TaskStatus.SUCCESS else "incomplete"

    @staticmethod
    def _artifact_is_mat(artifact: dict[str, str]) -> bool:
        kind = str(artifact.get("kind", "")).lower()
        path = str(artifact.get("path", "")).lower()
        return path.endswith(".mat") or "mat" in kind

    def _set_artifact_status_flags(self, task: Task) -> None:
        flag = self._task_status_flag(task.status)
        for artifact in task.artifacts:
            if self._artifact_is_mat(artifact):
                artifact["status_flag"] = flag

    def _write_task_status_flags_to_mat(self, task: Task) -> None:
        flag = self._task_status_flag(task.status)
        for artifact in task.artifacts:
            if not self._artifact_is_mat(artifact):
                continue
            path_text = artifact.get("path", "")
            if not path_text:
                continue
            try:
                mat_path = self._path_text_to_local(path_text)
            except Exception:
                continue
            if not mat_path.is_file():
                continue
            try:
                import scipy.io as sio

                payload = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=True)
                payload = {key: value for key, value in payload.items() if not key.startswith("__")}
                payload["TaskStatusFlag"] = flag
                payload["WebTaskStatusFlag"] = flag
                payload["WebTaskId"] = task.id
                sio.savemat(mat_path, payload, do_compression=True)
                artifact["status_flag"] = flag
            except Exception as exc:
                entry = {
                    "type": "info",
                    "text": f"无法写入 MAT status flag ({flag}): {mat_path} ({exc})",
                }
                task.logs.append(entry)
                self._append_log_file(task, entry)

    def _persist_task(self, task: Task) -> None:
        if not task.task_dir:
            return
        path = self._task_manifest_path(task)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(task.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _append_log_file(self, task: Task, entry: dict[str, str]) -> None:
        if not task.task_dir:
            return
        try:
            Path(task.task_dir).mkdir(parents=True, exist_ok=True)
            with self._task_log_jsonl_path(task).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            text_log = self.task_log_text_path(task)
            if not text_log.exists() or text_log.stat().st_size == 0:
                text_log.write_text("", encoding="utf-8-sig")
            with text_log.open("a", encoding="utf-8") as fh:
                fh.write(self._format_log_entry(entry))
        except OSError:
            pass

    @staticmethod
    def _format_log_entry(entry: dict[str, str]) -> str:
        return f"[{entry.get('type', 'info')}] {entry.get('text', '')}\n"

    def rewrite_task_log_text(self, task: Task) -> Path:
        path = self.task_log_text_path(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(self._format_log_entry(entry) for entry in task.logs)
        path.write_text(text, encoding="utf-8-sig")
        return path

    def _load_logs(self, task_dir: Path) -> list[dict[str, str]]:
        jsonl = task_dir / "logs.jsonl"
        if jsonl.is_file():
            logs: list[dict[str, str]] = []
            for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and "text" in item:
                    logs.append({"type": str(item.get("type", "info")), "text": str(item.get("text", ""))})
            return logs
        text_log = task_dir / "run.log"
        if text_log.is_file():
            return [
                {"type": "info", "text": line}
                for line in text_log.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            ]
        return []

    def _task_from_manifest(self, manifest_path: Path) -> Task | None:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            status = TaskStatus(str(data.get("status", TaskStatus.FAILED.value)))
        except ValueError:
            status = TaskStatus.FAILED
        task = Task(
            id=str(data.get("id") or manifest_path.parent.name),
            module=str(data.get("module") or ""),
            args=[str(item) for item in data.get("args") or []],
            status=status,
            created_at=str(data.get("created_at") or ""),
            started_at=str(data.get("started_at") or ""),
            finished_at=str(data.get("finished_at") or ""),
            exit_code=data.get("exit_code"),
            logs=self._load_logs(manifest_path.parent),
            artifacts=[
                self._normalize_artifact(dict(item))
                for item in data.get("artifacts") or []
                if isinstance(item, dict)
            ],
            error_summary=str(data.get("error_summary") or ""),
            task_dir=str(manifest_path.parent),
        )
        task.artifacts = self._recover_artifacts_from_logs(task.module, task.artifacts, task.logs)
        self._set_artifact_status_flags(task)
        return task

    def _load_persisted_tasks(self) -> None:
        for manifest in sorted(self.task_root.glob("*/manifest.json")):
            task = self._task_from_manifest(manifest)
            if task is not None:
                self._tasks[task.id] = task
                self._persist_task(task)

    def _task_sort_key(self, task: Task) -> str:
        return task.created_at or Path(task.task_dir).name

    def _prune_task_dirs(self) -> None:
        tasks = [
            task for task in self._tasks.values()
            if task.task_dir and Path(task.task_dir).is_dir()
        ]
        if len(tasks) <= MAX_PERSISTED_TASKS:
            return
        removable = sorted(
            [task for task in tasks if task.status not in {TaskStatus.RUNNING, TaskStatus.PENDING}],
            key=self._task_sort_key,
        )
        overflow = len(tasks) - MAX_PERSISTED_TASKS
        for task in removable[:overflow]:
            try:
                shutil.rmtree(task.task_dir)
            except OSError:
                continue
            self._tasks.pop(task.id, None)

    def _refresh_from_disk(self) -> None:
        existing_dirs = {path.parent.resolve() for path in self.task_root.glob("*/manifest.json")}
        for task_id, task in list(self._tasks.items()):
            if task.task_dir and Path(task.task_dir).resolve() not in existing_dirs and task.status not in {TaskStatus.RUNNING, TaskStatus.PENDING}:
                self._tasks.pop(task_id, None)
        self._load_persisted_tasks()
        self._prune_task_dirs()

    def create_task(self, module: str, args: list[str]) -> Task:
        task_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat(timespec="seconds")
        dir_stamp = created_at.replace(":", "").replace("-", "").replace("T", "_")
        task_dir = self.task_root / f"{dir_stamp}_{task_id}_{self._slug(module)}"
        task = Task(
            id=task_id,
            module=module,
            args=args,
            created_at=created_at,
            task_dir=str(task_dir),
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        self._tasks[task.id] = task
        self._prepend_config_log(task)
        self._persist_task(task)
        self._prune_task_dirs()
        return task

    def _config_path_from_args(self, args: list[str]) -> Path | None:
        def resolve_candidate(text: str) -> Path | None:
            try:
                path = self._path_text_to_local(text)
            except Exception:
                return None
            if path.is_file():
                return path
            if path.is_dir():
                for workflow in (
                    "rsdm",
                    "rsdm_shakedown",
                    "direct_methods_steady_state",
                    "direct_methods_steady_state_rsdm",
                    "direct_methods_steady_state_dca",
                    "dca",
                    "shakedown",
                    "inc_analysis",
                    "validation",
                ):
                    try:
                        resolved = resolve_workflow_target(path, workflow)
                    except Exception:
                        continue
                    cfg_path = resolved.get("config_path")
                    if cfg_path is not None and Path(cfg_path).is_file():
                        return Path(cfg_path).resolve()
                cfg_files = sorted(path.glob("*.cfg"))
                if cfg_files:
                    return cfg_files[0].resolve()
            return path

        for idx, arg in enumerate(args):
            if str(arg) == "--config" and idx + 1 < len(args):
                return resolve_candidate(str(args[idx + 1]))
        for arg in args:
            text = str(arg)
            if text.lower().endswith(".cfg") or text.endswith("/") or text.endswith("\\"):
                resolved = resolve_candidate(text)
                if resolved is not None:
                    return resolved
        return None

    def _prepend_config_log(self, task: Task) -> None:
        cfg_path = self._config_path_from_args(task.args)
        if cfg_path is None:
            return
        if not cfg_path.is_file():
            entry = {"type": "info", "text": f"===== CONFIG 未找到 =====\n{cfg_path}"}
        else:
            try:
                content = cfg_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                content = f"<读取失败: {exc}>"
            entry = {"type": "info", "text": f"===== CONFIG: {cfg_path} =====\n{content.rstrip()}\n===== END CONFIG ====="}
        task.logs.insert(0, entry)
        self._append_log_file(task, entry)

    def task_config_snapshot(self, task: Task) -> dict[str, str]:
        cfg_path = self._config_path_from_args(task.args)
        if cfg_path is None:
            return {"path": "", "content": "", "exists": "false"}
        if not cfg_path.is_file():
            return {"path": str(cfg_path), "content": "", "exists": "false"}
        try:
            content = cfg_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            content = f"<读取失败: {exc}>"
        return {"path": str(cfg_path), "content": content, "exists": "true"}

    def get_task(self, task_id: str) -> Optional[Task]:
        self._refresh_from_disk()
        return self._tasks.get(task_id)

    def list_tasks(self, limit: int = 20) -> list[dict]:
        self._refresh_from_disk()
        items = sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True,
        )[:limit]
        return [t.to_dict() for t in items]

    def delete_tasks(self, task_ids: list[str]) -> dict[str, Any]:
        self._refresh_from_disk()
        deleted: list[str] = []
        skipped: list[dict[str, str]] = []
        root = self.task_root.resolve()
        for raw_id in task_ids:
            task_id = str(raw_id or "").strip()
            if not task_id:
                continue
            task = self._tasks.get(task_id)
            if task is None:
                skipped.append({"id": task_id, "reason": "not_found"})
                continue
            if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
                skipped.append({"id": task_id, "reason": "running"})
                continue
            if task.task_dir:
                try:
                    folder = Path(task.task_dir).resolve()
                    if folder == root or root not in folder.parents:
                        skipped.append({"id": task_id, "reason": "outside_task_root"})
                        continue
                    if folder.is_dir():
                        shutil.rmtree(folder)
                except OSError as exc:
                    skipped.append({"id": task_id, "reason": str(exc)})
                    continue
            self._tasks.pop(task_id, None)
            deleted.append(task_id)
        return {"ok": True, "deleted": deleted, "skipped": skipped}

    async def stop_task(self, task_id: str) -> dict[str, Any]:
        task = self._tasks.get(task_id) or self.get_task(task_id)
        if task is None:
            return {"ok": False, "error": "Task not found"}
        if task.status not in {TaskStatus.PENDING, TaskStatus.RUNNING}:
            return {"ok": True, "status": task.status.value, "message": "Task is not running"}

        self._stop_requested.add(task_id)
        entry = {"type": "info", "text": "收到停止请求：将终止当前计算进程，并保留已经写出的 MAT/artifact。"}
        task.logs.append(entry)
        self._append_log_file(task, entry)
        await self._push(task.id, entry)

        proc = self._active_processes.get(task_id)
        if proc is None:
            task.status = TaskStatus.STOPPED
            task.exit_code = -15
            task.finished_at = datetime.now().isoformat(timespec="seconds")
            task.error_summary = "Task stopped before subprocess started."
            self._recover_artifacts_near_config(task)
            self._set_artifact_status_flags(task)
            self._write_task_status_flags_to_mat(task)
            self._persist_task(task)
            await self._push(task.id, {
                "type": "status",
                "status": task.status.value,
                "exit_code": task.exit_code,
                "error_summary": task.error_summary,
                "artifacts": task.artifacts,
            })
            return {"ok": True, "status": task.status.value}

        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        return {"ok": True, "status": "stopping"}

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
    def _artifact_label_for(module: str, path: str) -> str:
        module_text = str(module or "").lower()
        path_text = str(path or "").lower()
        if "steady_state_dca" in path_text or "jax_dca" in path_text or ".dca" in module_text:
            return "DCA result MAT"
        if "steady_state_rsdm" in path_text or "jax_rsdm" in path_text:
            return "RSDM result MAT"
        if "shakedown" in module_text:
            return "Shakedown result MAT"
        if "validation" in module_text:
            return f"Validation result MAT: {Path(path).stem}"
        if "direct_methods" in module_text:
            return "Direct Methods result MAT"
        return Path(path).name

    @classmethod
    def _parse_artifact_line(cls, line: str, module: str = "") -> dict[str, str] | None:
        text = str(line or "").strip()
        prefix = "__ARTIFACT__|"
        if text.startswith(prefix):
            payload = text[len(prefix):]
            parts = payload.split("|", 2)
            if len(parts) < 2:
                return None
            kind = parts[0].strip() or "file"
            path = parts[1].strip()
            if not path:
                return None
            label = parts[2].strip() if len(parts) >= 3 else Path(path).name
            return cls._normalize_artifact({"kind": kind, "path": path, "label": label})

        match = re.search(r"\boutput:\s+(.+?\.mat)\s*$", text, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"\bExported MAT:\s+(.+?\.mat)\s*$", text, flags=re.IGNORECASE)
        if not match and "validation" in str(module or "").lower():
            match = re.search(
                r"(?:validation\s+MAT\s+.*?(?:已写入|written\s+to)|结果\s+MAT\s+已放入)\s+(.+?\.mat)\s*$",
                text,
                flags=re.IGNORECASE,
            )
        if not match:
            return None
        path = match.group(1).strip().strip("\"'")
        if not path:
            return None
        return cls._normalize_artifact({
            "kind": "mat",
            "path": path,
            "label": cls._artifact_label_for(module, path),
        })

    def _recover_artifacts_from_logs(
        self,
        module: str,
        artifacts: list[dict[str, str]],
        logs: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        recovered: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add(item: dict[str, str] | None) -> None:
            if not item:
                return
            artifact = self._normalize_artifact(item)
            path = artifact.get("path", "")
            if not path:
                return
            existing = next((entry for entry in recovered if entry.get("path") == path), None)
            if existing is not None:
                existing.update(artifact)
                return
            key = (artifact.get("kind", ""), path)
            if key in seen:
                return
            seen.add(key)
            recovered.append(artifact)

        for item in artifacts:
            add(item)
        for entry in logs:
            add(self._parse_artifact_line(str(entry.get("text", "")), module))
        return recovered

    def _recover_artifacts_near_config(self, task: Task) -> None:
        cfg_path = self._config_path_from_args(task.args)
        if cfg_path is None or not cfg_path.parent.is_dir():
            return
        try:
            started = datetime.fromisoformat(task.started_at) if task.started_at else datetime.fromisoformat(task.created_at)
            threshold = started.timestamp() - 2.0
        except Exception:
            threshold = 0.0
        for mat_path in sorted(cfg_path.parent.glob("*.mat"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0):
            try:
                if mat_path.stat().st_mtime < threshold:
                    continue
            except OSError:
                continue
            artifact = self._normalize_artifact({
                "kind": "mat",
                "path": str(mat_path),
                "label": self._artifact_label_for(task.module, str(mat_path)),
            })
            if artifact.get("path") and not any(item.get("path") == artifact["path"] for item in task.artifacts):
                task.artifacts.append(artifact)

    async def run_task(self, task: Task) -> None:
        async with self._semaphore:
            if task.id in self._stop_requested:
                task.status = TaskStatus.STOPPED
                task.exit_code = -15
                task.finished_at = datetime.now().isoformat(timespec="seconds")
                task.error_summary = "Task stopped before subprocess started."
                self._persist_task(task)
                await self._push(task.id, {
                    "type": "status",
                    "status": task.status.value,
                    "exit_code": task.exit_code,
                    "error_summary": task.error_summary,
                    "artifacts": task.artifacts,
                })
                return
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat(timespec="seconds")
            self._persist_task(task)
            await self._push(task.id, {"type": "status", "status": "running"})

            async def on_stdout(line: str):
                artifact = self._parse_artifact_line(line, task.module)
                if artifact is not None:
                    existing = next(
                        (item for item in task.artifacts if item.get("path") == artifact["path"]),
                        None,
                    )
                    if existing is not None:
                        existing.update(artifact)
                    else:
                        task.artifacts.append(artifact)
                    await self._push(task.id, {"type": "artifact", "artifact": artifact})
                    info_entry = {"type": "info", "text": f"生成结果产物: {artifact['label']}"}
                    task.logs.append(info_entry)
                    self._append_log_file(task, info_entry)
                    self._persist_task(task)
                    await self._push(task.id, info_entry)
                    if str(line or "").strip().startswith("__ARTIFACT__|"):
                        return
                entry = {"type": "stdout", "text": line}
                task.logs.append(entry)
                self._append_log_file(task, entry)
                await self._push(task.id, entry)

            async def on_stderr(line: str):
                entry = {"type": "stderr", "text": line}
                task.logs.append(entry)
                self._append_log_file(task, entry)
                await self._push(task.id, entry)

            try:
                def on_process(proc):
                    self._active_processes[task.id] = proc

                exit_code = await run_wsl_module_async(
                    task.module, task.args,
                    on_stdout=on_stdout, on_stderr=on_stderr, on_process=on_process,
                )
                task.exit_code = exit_code
                if task.id in self._stop_requested:
                    task.status = TaskStatus.STOPPED
                    task.error_summary = "Task stopped by user."
                else:
                    task.status = TaskStatus.SUCCESS if exit_code == 0 else TaskStatus.FAILED
                if exit_code != 0 and task.status != TaskStatus.STOPPED:
                    # Extract last few stderr lines as error summary
                    stderr_lines = [
                        e["text"] for e in task.logs if e["type"] == "stderr"
                    ]
                    task.error_summary = "\n".join(stderr_lines[-5:])
            except Exception as exc:
                task.status = TaskStatus.STOPPED if task.id in self._stop_requested else TaskStatus.FAILED
                task.error_summary = "Task stopped by user." if task.status == TaskStatus.STOPPED else str(exc)
                entry = {"type": "stderr", "text": str(exc)}
                task.logs.append(entry)
                self._append_log_file(task, entry)
            finally:
                self._active_processes.pop(task.id, None)

            task.finished_at = datetime.now().isoformat(timespec="seconds")
            task.artifacts = self._recover_artifacts_from_logs(task.module, task.artifacts, task.logs)
            self._recover_artifacts_near_config(task)
            self._set_artifact_status_flags(task)
            self._write_task_status_flags_to_mat(task)
            self._persist_task(task)
            await self._push(task.id, {
                "type": "status",
                "status": task.status.value,
                "exit_code": task.exit_code,
                "error_summary": task.error_summary,
                "artifacts": task.artifacts,
            })


# Singleton
task_manager = TaskManager()
