"""Settings routes — MCP connection configuration."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..job_store import JobStatus, list_jobs
from ..mcp_client import disconnect, get_mcp_settings, update_mcp_settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


class MCPSettingsResponse(BaseModel):
    mode: str
    url: str | None
    server_path: str | None
    has_auth: bool
    output_folder: str | None


class MCPSettingsUpdate(BaseModel):
    mode: Literal["local", "remote"] | None = None
    url: str | None = None
    server_path: str | None = None
    auth_token: str | None = None
    output_folder: str | None = None

    model_config = {"protected_namespaces": ()}


@router.get("", response_model=MCPSettingsResponse)
async def get_settings():
    s = get_mcp_settings()
    return MCPSettingsResponse(
        mode=s.mode,
        url=s.url,
        server_path=s.server_path,
        has_auth=s.auth_token is not None,
        output_folder=s.output_folder,
    )


@router.put("", response_model=MCPSettingsResponse)
async def put_settings(req: MCPSettingsUpdate):
    connection_changed = any(
        x is not None for x in (req.mode, req.url, req.server_path, req.auth_token)
    )
    if connection_changed:
        running = [j for j in list_jobs() if j.status in (JobStatus.PENDING, JobStatus.RUNNING)]
        if running:
            raise HTTPException(
                409, f"Cannot change connection settings while {len(running)} job(s) are still running."
            )
        await disconnect()
    s = update_mcp_settings(
        mode=req.mode,
        url=req.url,
        server_path=req.server_path,
        auth_token=req.auth_token,
        output_folder=req.output_folder,
    )
    return MCPSettingsResponse(
        mode=s.mode,
        url=s.url,
        server_path=s.server_path,
        has_auth=s.auth_token is not None,
        output_folder=s.output_folder,
    )
