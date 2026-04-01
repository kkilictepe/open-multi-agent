"""Built-in grep tool — searches for a regex pattern in files."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from ...types import ToolResult, ToolUseContext
from ..framework import define_tool

DEFAULT_MAX_RESULTS = 100
SKIP_DIRS = {".git", ".svn", ".hg", "node_modules", ".next", "dist", "build"}


class GrepInput(BaseModel):
    pattern: str = Field(description="Regular expression pattern to search for in file contents.")
    path: str | None = Field(
        default=None,
        description="Directory or file path to search in. Defaults to the current working directory.",
    )
    glob: str | None = Field(
        default=None,
        description='Glob pattern to filter which files are searched (e.g. "*.ts", "**/*.json").',
    )
    max_results: int | None = Field(
        default=None,
        description=f"Maximum number of matching lines to return. Defaults to {DEFAULT_MAX_RESULTS}.",
    )


async def _execute(input: GrepInput, context: ToolUseContext) -> ToolResult:
    search_path = input.path or os.getcwd()
    max_results = input.max_results or DEFAULT_MAX_RESULTS

    try:
        regex = re.compile(input.pattern)
    except re.error:
        return ToolResult(data=f'Invalid regular expression: "{input.pattern}"', is_error=True)

    # Try ripgrep first
    if _is_ripgrep_available():
        return await _run_ripgrep(
            input.pattern, search_path, glob=input.glob, max_results=max_results,
            cancel_event=context.cancel_event,
        )

    # Fallback: pure Python recursive search
    return await _run_python_search(
        regex, search_path, glob=input.glob, max_results=max_results,
        cancel_event=context.cancel_event,
    )


def _is_ripgrep_available() -> bool:
    return shutil.which("rg") is not None


async def _run_ripgrep(
    pattern: str,
    search_path: str,
    *,
    glob: str | None,
    max_results: int,
    cancel_event: asyncio.Event | None,
) -> ToolResult:
    args = [
        "rg",
        "--line-number",
        "--no-heading",
        "--color=never",
        f"--max-count={max_results}",
    ]
    if glob is not None:
        args.extend(["--glob", glob])
    args.extend(["--", pattern, search_path])

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
    except Exception:
        return ToolResult(
            data="ripgrep process error \u2014 run may be retried with the Python fallback.",
            is_error=True,
        )

    # rg exit code 1 = no matches (not an error)
    if proc.returncode not in (0, 1):
        err_msg = stderr_bytes.decode("utf-8", errors="replace").strip() if stderr_bytes else ""
        return ToolResult(
            data=f"ripgrep failed (exit {proc.returncode}): {err_msg}", is_error=True
        )

    output = stdout_bytes.decode("utf-8", errors="replace").rstrip() if stdout_bytes else ""
    if not output:
        return ToolResult(data="No matches found.", is_error=False)

    return ToolResult(data=output, is_error=False)


async def _run_python_search(
    regex: re.Pattern[str],
    search_path: str,
    *,
    glob: str | None,
    max_results: int,
    cancel_event: asyncio.Event | None,
) -> ToolResult:
    p = Path(search_path)
    try:
        if p.is_file():
            files = [p]
        else:
            files = await asyncio.to_thread(_collect_files, p, glob)
    except Exception as e:
        return ToolResult(data=f'Cannot access path "{search_path}": {e}', is_error=True)

    matches: list[str] = []
    cwd = os.getcwd()

    for file in files:
        if cancel_event and cancel_event.is_set():
            break
        if len(matches) >= max_results:
            break

        try:
            content = await asyncio.to_thread(file.read_text, encoding="utf-8")
        except Exception:
            continue

        for i, line in enumerate(content.split("\n")):
            if len(matches) >= max_results:
                break
            if regex.search(line):
                try:
                    rel = os.path.relpath(str(file), cwd)
                except ValueError:
                    rel = str(file)
                matches.append(f"{rel}:{i + 1}:{line}")

    if not matches:
        return ToolResult(data="No matches found.", is_error=False)

    formatted = "\n".join(matches)
    truncation_note = (
        f"\n\n(results capped at {max_results}; use max_results to raise the limit)"
        if len(matches) >= max_results
        else ""
    )

    return ToolResult(data=formatted + truncation_note, is_error=False)


def _collect_files(directory: Path, glob_pattern: str | None) -> list[Path]:
    results: list[Path] = []
    _walk(directory, glob_pattern, results)
    return results


def _walk(directory: Path, glob_pattern: str | None, results: list[Path]) -> None:
    try:
        entries = list(directory.iterdir())
    except PermissionError:
        return

    for entry in entries:
        if entry.is_dir():
            if entry.name not in SKIP_DIRS:
                _walk(entry, glob_pattern, results)
        elif entry.is_file():
            if glob_pattern is None or _matches_glob(entry.name, glob_pattern):
                results.append(entry)


def _matches_glob(filename: str, glob_pattern: str) -> bool:
    pattern = glob_pattern.removeprefix("**/")
    return fnmatch.fnmatch(filename.lower(), pattern.lower())


grep_tool = define_tool(
    name="grep",
    description=(
        "Search for a regular-expression pattern in one or more files. "
        "Returns matching lines with their file paths and 1-based line numbers. "
        "Use the `glob` parameter to restrict the search to specific file types "
        '(e.g. "*.ts"). '
        "Results are capped by `max_results` to keep the response manageable."
    ),
    input_schema=GrepInput,
    execute=_execute,
)
