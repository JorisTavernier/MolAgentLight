"""Admin routes — token management via MCP admin_manage tool."""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..mcp_client import call_tool

router = APIRouter(prefix="/api/admin", tags=["admin"])


class AdminManageRequest(BaseModel):
    action: Literal["create_token", "revoke_user", "rotate_token", "list_users"]
    user_id: Optional[str] = None
    owner_id: Optional[str] = None


@router.post("/manage")
async def admin_manage(req: AdminManageRequest):
    """Proxy to MCP admin_manage tool. Uses the configured MCP auth token."""
    args = req.model_dump(exclude_none=True)
    try:
        result = await call_tool("admin_manage", args)
    except RuntimeError as exc:
        raise HTTPException(500, f"Admin operation failed: {exc}")
    return result
