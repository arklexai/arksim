# SPDX-License-Identifier: Apache-2.0
"""Exposes the ADK tools over MCP (streamable HTTP), port 9997.

Uses the low-level mcp.server.Server directly, not the decorator-per-tool
MCPServer - the point is to demonstrate the exact wrapping path a partner
with existing FunctionTools would take: wrap what
already exists as MCP tools, don't redeclare them in a new framework's
decorator style.

Run:
    python -m google_adk.mcp_server

Environment:
    MCP_HOST, MCP_PORT       bind address (default 127.0.0.1:9997)
    MCP_ALLOWED_HOSTS        comma-separated Host values to accept, needed
                             when serving through a tunnel or proxy
    MCP_AUTH_TOKEN           require `Authorization: Bearer <token>`
    MCP_AUTH_HEADER          require a custom header, as `Name: value`
    MCP_FAULT                down | slow:<ms> | fail_every:<n> | unauthorized
    MCP_FAULT_TOOLS          comma-separated tools that always fail
"""

from __future__ import annotations

import os
import sys
from itertools import count
from pathlib import Path

import anyio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mcp.types as mcp_types  # noqa: E402
import uvicorn  # noqa: E402
from adk_tools import MCP_TOOLS, TOOL_ANNOTATIONS  # noqa: E402
from google.adk.tools.mcp_tool.conversion_utils import (
    adk_to_mcp_tool_type,  # noqa: E402
)
from mcp.server import Server  # noqa: E402
from mcp.server.transport_security import (
    TransportSecuritySettings,  # noqa: E402
)
from mcp_shapes import build_result, error_result  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.types import Receive, Scope, Send  # noqa: E402
from store_tools import init_db  # noqa: E402

server = Server("customer-service-knowledge")

_TOOLS_BY_NAME = {tool.name: tool for tool in MCP_TOOLS}
_FAULT_TOOLS = {
    name.strip() for name in os.getenv("MCP_FAULT_TOOLS", "").split(",") if name.strip()
}
_request_counter = count(1)


async def list_tools(
    ctx: object, params: mcp_types.PaginatedRequestParams
) -> mcp_types.ListToolsResult:
    tools = []
    for name, adk_tool in _TOOLS_BY_NAME.items():
        mcp_tool = adk_to_mcp_tool_type(adk_tool)
        # ADK emits an empty input schema for a zero-argument tool, which MCP
        # rejects: inputSchema.type is required. Any partner wrapping an
        # existing FunctionTool set hits this, so normalize it here.
        if not mcp_tool.input_schema.get("type"):
            mcp_tool.input_schema = {"type": "object", "properties": {}}
        annotation = TOOL_ANNOTATIONS.get(name)
        if annotation:
            mcp_tool.annotations = mcp_types.ToolAnnotations(**annotation)
        tools.append(mcp_tool)
    return mcp_types.ListToolsResult(tools=tools)


async def call_tool(
    ctx: object, params: mcp_types.CallToolRequestParams
) -> mcp_types.CallToolResult:
    name = params.name
    adk_tool = _TOOLS_BY_NAME.get(name)
    if adk_tool is None:
        return error_result(f"Unknown tool: {name}")
    if name in _FAULT_TOOLS:
        return error_result(f"Injected failure for tool '{name}' (MCP_FAULT_TOOLS)")
    try:
        # tool_context=None is safe here: none of store_tools.py's functions
        # declare a tool_context parameter, so ADK never tries to pass it through.
        result = await adk_tool.run_async(
            args=params.arguments or {}, tool_context=None
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as isError
        return error_result(f"{type(exc).__name__}: {exc}")
    return build_result(name, str(result))


server.add_request_handler("tools/list", mcp_types.PaginatedRequestParams, list_tools)
server.add_request_handler("tools/call", mcp_types.CallToolRequestParams, call_tool)


def _transport_security() -> TransportSecuritySettings | None:
    """Allow the Host values a tunnel or proxy will present, when configured."""
    hosts = [
        h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()
    ]
    if not hosts:
        return None
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=[f"https://{h}" for h in hosts]
        + [f"http://{h}" for h in hosts],
    )


class _Gate:
    """Auth and fault injection in front of the MCP transport.

    The platform connector needs a target that can reject bad credentials,
    stall, and fail, so those behaviours are switched on by environment
    variables rather than requiring a second server implementation.
    """

    def __init__(self, app: object) -> None:
        self.app = app
        self.token = os.getenv("MCP_AUTH_TOKEN", "")
        self.fault = os.getenv("MCP_FAULT", "")
        header = os.getenv("MCP_AUTH_HEADER", "")
        self.header_name, _, self.header_value = header.partition(":")
        self.header_name = self.header_name.strip().lower()
        self.header_value = self.header_value.strip()

    async def _deny(
        self, scope: Scope, receive: Receive, send: Send, status: int, message: str
    ) -> None:
        await JSONResponse({"error": message}, status_code=status)(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        mode, _, argument = self.fault.partition(":")

        if mode == "down":
            await self._deny(scope, receive, send, 503, "Knowledge source unavailable")
            return
        if mode == "unauthorized":
            await self._deny(scope, receive, send, 401, "Invalid credentials")
            return
        if mode == "fail_every" and next(_request_counter) % int(argument) == 0:
            await self._deny(scope, receive, send, 500, "Injected upstream failure")
            return
        if mode == "slow":
            await anyio.sleep(int(argument) / 1000)

        if self.token and headers.get("authorization") != f"Bearer {self.token}":
            await self._deny(
                scope, receive, send, 401, "Invalid or missing bearer token"
            )
            return
        if self.header_name and headers.get(self.header_name) != self.header_value:
            await self._deny(
                scope,
                receive,
                send,
                401,
                f"Invalid or missing {self.header_name} header",
            )
            return

        await self.app(scope, receive, send)


def build_app() -> object:
    return _Gate(
        server.streamable_http_app(
            streamable_http_path="/mcp",
            stateless_http=True,
            host=os.getenv("MCP_HOST", "127.0.0.1"),
            transport_security=_transport_security(),
        )
    )


def main() -> None:
    init_db()
    uvicorn.run(
        build_app(),
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "9997")),
    )


if __name__ == "__main__":
    main()
