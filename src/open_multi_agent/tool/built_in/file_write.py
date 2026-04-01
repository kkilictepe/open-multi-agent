"""Built-in file-write tool — creates or overwrites a file."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from ...types import ToolResult, ToolUseContext
from ..framework import define_tool


class FileWriteInput(BaseModel):
    path: str = Field(
        description="Absolute path to the file to write. The path must be absolute."
    )
    content: str = Field(description="The full content to write to the file.")


async def _execute(input: FileWriteInput, context: ToolUseContext) -> ToolResult:
    file_path = Path(input.path)
    existed = file_path.exists()

    # Ensure parent directory hierarchy exists
    parent_dir = file_path.parent
    try:
        await asyncio.to_thread(parent_dir.mkdir, parents=True, exist_ok=True)
    except Exception as e:
        return ToolResult(
            data=f'Failed to create parent directory "{parent_dir}": {e}', is_error=True
        )

    # Write the file
    try:
        await asyncio.to_thread(file_path.write_text, input.content, encoding="utf-8")
    except Exception as e:
        return ToolResult(data=f'Failed to write file "{input.path}": {e}', is_error=True)

    line_count = len(input.content.split("\n"))
    byte_count = len(input.content.encode("utf-8"))
    action = "Updated" if existed else "Created"
    s = "" if line_count == 1 else "s"

    return ToolResult(
        data=f'{action} "{input.path}" ({line_count} line{s}, {byte_count} bytes).',
        is_error=False,
    )


file_write_tool = define_tool(
    name="file_write",
    description=(
        "Write content to a file, creating it (and any missing parent directories) if it "
        "does not already exist, or overwriting it if it does. "
        "Prefer this tool for creating new files; use file_edit for targeted in-place edits "
        "of existing files."
    ),
    input_schema=FileWriteInput,
    execute=_execute,
)
