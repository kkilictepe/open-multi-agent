"""Built-in file-edit tool — targeted string replacement in an existing file."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from ...types import ToolResult, ToolUseContext
from ..framework import define_tool


class FileEditInput(BaseModel):
    path: str = Field(description="Absolute path to the file to edit.")
    old_string: str = Field(
        description="The exact string to find and replace. Must match character-for-character including whitespace and newlines."
    )
    new_string: str = Field(
        description="The replacement string that will be inserted in place of `old_string`."
    )
    replace_all: bool | None = Field(
        default=None,
        description="When true, replace every occurrence of `old_string` instead of requiring it to be unique. Defaults to false.",
    )


async def _execute(input: FileEditInput, context: ToolUseContext) -> ToolResult:
    try:
        original = await asyncio.to_thread(Path(input.path).read_text, encoding="utf-8")
    except Exception as e:
        return ToolResult(data=f'Could not read "{input.path}": {e}', is_error=True)

    occurrences = original.count(input.old_string) if input.old_string else 0

    if occurrences == 0:
        return ToolResult(
            data=(
                f'The string to replace was not found in "{input.path}".\n'
                "Make sure `old_string` matches the file contents exactly, "
                "including indentation and line endings."
            ),
            is_error=True,
        )

    do_replace_all = input.replace_all or False

    if occurrences > 1 and not do_replace_all:
        return ToolResult(
            data=(
                f'`old_string` appears {occurrences} times in "{input.path}". '
                "Provide a more specific string to uniquely identify the section you want "
                "to replace, or set `replace_all: true` to replace every occurrence."
            ),
            is_error=True,
        )

    # Perform the replacement
    if do_replace_all:
        updated = original.replace(input.old_string, input.new_string)
    else:
        updated = original.replace(input.old_string, input.new_string, 1)

    try:
        await asyncio.to_thread(Path(input.path).write_text, updated, encoding="utf-8")
    except Exception as e:
        return ToolResult(data=f'Failed to write "{input.path}": {e}', is_error=True)

    replaced_count = occurrences if do_replace_all else 1
    s = "" if replaced_count == 1 else "s"
    return ToolResult(
        data=f'Replaced {replaced_count} occurrence{s} in "{input.path}".',
        is_error=False,
    )


file_edit_tool = define_tool(
    name="file_edit",
    description=(
        "Edit a file by replacing a specific string with new content. "
        "The `old_string` must appear verbatim in the file. "
        "By default the tool errors if `old_string` appears more than once \u2014 "
        "use `replace_all: true` to replace every occurrence. "
        "Use file_write when you need to create a new file or rewrite it entirely."
    ),
    input_schema=FileEditInput,
    execute=_execute,
)
