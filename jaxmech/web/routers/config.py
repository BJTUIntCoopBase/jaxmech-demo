"""Configuration API — read / update env.cfg and module-level config files."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from jaxmech.web.services.config_manager import (
    read_env_cfg,
    read_module_cfg,
    write_env_cfg,
    write_module_cfg,
)

router = APIRouter(prefix="/api/config", tags=["config"])


class EnvUpdate(BaseModel):
    values: dict[str, str]


class ModuleCfgUpdate(BaseModel):
    content: str


@router.get("/env")
async def get_env_config():
    return read_env_cfg()


@router.put("/env")
async def update_env_config(body: EnvUpdate):
    write_env_cfg(body.values)
    return {"ok": True}


@router.get("/{module}")
async def get_module_config(module: str, model_path: str):
    return read_module_cfg(module, model_path)


@router.put("/{module}")
async def update_module_config(module: str, model_path: str, body: ModuleCfgUpdate):
    path = write_module_cfg(module, model_path, body.content)
    return {"ok": True, "path": path}
