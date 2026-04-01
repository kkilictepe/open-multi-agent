"""Built-in file-read tool — reads a file from disk with line numbers."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from ...types import ToolResult, ToolUseContext
from ..framework import define_tool


class FileReadInput(BaseModel):
    path: str = Field(description="Absolute path to the file to read.")
    offset: int | None = Field(
        default=None,
        description="1-based line number to start reading from. When omitted the file is read from the beginning.",
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of lines to return. When omitted all lines from offset to the end are returned.",
    )


async def _execute(input: FileReadInput, context: ToolUseContext) -> ToolResult:
    try:
        raw = await asyncio.to_thread(Path(input.path).read_text, encoding="utf-8")
    except Exception as e:
        return ToolResult(data=f'Could not read file "{input.path}": {e}', is_error=True)

    lines = raw.split("\n")
    # Remove the last empty string produced by a trailing newline
    if lines and lines[-1] == "":
        lines.pop()

    total_lines = len(lines)

    # Apply offset (convert from 1-based to 0-based)
    start_index = max(0, input.offset - 1) if input.offset is not None else 0

    if start_index >= total_lines and total_lines > 0:
        s = "" if total_lines == 1 else "s"
        return ToolResult(
            data=f'File "{input.path}" has {total_lines} line{s} but offset {input.offset} is beyond the end.',
            is_error=True,
        )

    end_index = (
        min(start_index + input.limit, total_lines) if input.limit is not None else total_lines
    )

    sliced = lines[start_index:end_index]

    numbered = "\n".join(f"{start_index + i + 1}\t{line}" for i, line in enumerate(sliced))

    meta = (
        f"\n\n(showing lines {start_index + 1}\u2013{end_index} of {total_lines})"
        if end_index < total_lines
        else ""
    )

    return ToolResult(data=numbered + meta, is_error=False)


file_read_tool = define_tool(
    name="file_read",
    description=(
        "Read the contents of a file from disk. "
        'Returns the file contents with line numbers prefixed in the format "N\\t<line>". '
        "Use `offset` and `limit` to read large files in chunks without loading the "
        "entire file into the context window."
    ),
    input_schema=FileReadInput,
    execute=_execute,
)
