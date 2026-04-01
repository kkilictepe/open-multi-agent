"""Built-in bash tool — executes a shell command and returns stdout + stderr."""

from __future__ import annotations

import asyncio
import sys

from pydantic import BaseModel, Field

from ...types import ToolResult, ToolUseContext
from ..framework import define_tool

DEFAULT_TIMEOUT_MS = 30_000


class BashInput(BaseModel):
    command: str = Field(description="The bash command to execute.")
    timeout: int | None = Field(
        default=None,
        description=f"Timeout in milliseconds before the command is forcibly killed. Defaults to {DEFAULT_TIMEOUT_MS} ms.",
    )
    cwd: str | None = Field(
        default=None, description="Working directory in which to run the command."
    )


async def _execute(input: BashInput, context: ToolUseContext) -> ToolResult:
    timeout_ms = input.timeout or DEFAULT_TIMEOUT_MS

    stdout, stderr, exit_code = await _run_command(
        input.command,
        cwd=input.cwd,
        timeout_ms=timeout_ms,
        cancel_event=context.cancel_event,
    )

    combined = _build_output(stdout, stderr, exit_code)
    return ToolResult(data=combined, is_error=exit_code != 0)


async def _run_command(
    command: str,
    *,
    cwd: str | None,
    timeout_ms: int,
    cancel_event: asyncio.Event | None,
) -> tuple[str, str, int]:
    try:
        if sys.platform == "win32":
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
    except Exception as e:
        return "", str(e), 127

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_ms / 1000.0
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return "", "Command timed out", 124

    return (
        stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else "",
        stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else "",
        proc.returncode or 0,
    )


def _build_output(stdout: str, stderr: str, exit_code: int) -> str:
    parts: list[str] = []

    if stdout:
        parts.append(stdout)

    if stderr:
        parts.append(f"--- stderr ---\n{stderr}" if stdout else stderr)

    if not parts:
        return (
            "(command completed with no output)"
            if exit_code == 0
            else f"(command exited with code {exit_code}, no output)"
        )

    if exit_code != 0:
        parts.append(f"\n(exit code: {exit_code})")

    return "\n".join(parts)


bash_tool = define_tool(
    name="bash",
    description=(
        "Execute a bash command and return its stdout and stderr. "
        "Use this for file system operations, running scripts, installing packages, "
        "and any task that requires shell access. "
        "The command runs in a non-interactive shell (bash -c). "
        "Long-running commands should use the timeout parameter."
    ),
    input_schema=BashInput,
    execute=_execute,
)
