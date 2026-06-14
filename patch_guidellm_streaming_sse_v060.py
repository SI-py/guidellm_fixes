#!/usr/bin/env python3
"""
Patch GuideLLM 0.6.x OpenAI HTTP streaming/SSE handling.

What it changes:
1) guidellm/backends/openai/http.py
   - Replaces OpenAIHTTPBackend._aiter_lines() so it parses Server-Sent Events
     by SSE event boundary (blank line) instead of httpx.aiter_lines().
   - This helps GPT-OSS /v1/completions streams where text chunks may contain
     newline characters that can break line-based parsing.

2) guidellm/backends/openai/request_handlers.py
   - Replaces TextCompletionsRequestHandler.extract_line_data() with a safer
     version that detects {"error": ...} SSE payloads and raises a clear error.
   - ChatCompletionsRequestHandler inherits this method, so chat streams benefit too.

Usage:
  python patch_guidellm_streaming_sse_v060.py
  python patch_guidellm_streaming_sse_v060.py --rollback
  python patch_guidellm_streaming_sse_v060.py --dry-run

Backups:
  *.bak_streaming_sse_v060 files are created next to patched files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BACKUP_SUFFIX = ".bak_streaming_sse_v060"


def find_guidellm_root() -> Path:
    try:
        import guidellm  # type: ignore
    except ImportError as exc:
        raise RuntimeError("guidellm is not installed in this Python environment") from exc

    return Path(guidellm.__file__).resolve().parent


def backup_file(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[backup] {path} -> {backup}")
    else:
        print(f"[backup] already exists: {backup}")
    return backup


def rollback_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        print(f"[rollback] no backup for {path}")
        return
    shutil.copy2(backup, path)
    print(f"[rollback] restored {path}")


def find_indented_block(lines: list[str], start_idx: int, indent: str) -> tuple[int, int]:
    """Return [start, end) for a method/class block starting at start_idx.

    The block ends at the first non-empty line with indentation <= block indent,
    excluding decorators immediately preceding the next block.
    """
    start = start_idx
    end = len(lines)
    base_len = len(indent)

    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            continue

        leading = len(line) - len(line.lstrip(" "))
        if leading <= base_len:
            end = i
            break

    return start, end


def replace_method_in_class(
    text: str,
    class_name: str,
    method_name: str,
    new_method: str,
) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)

    class_idx = None
    for i, line in enumerate(lines):
        if line.startswith(f"class {class_name}"):
            class_idx = i
            break

    if class_idx is None:
        raise RuntimeError(f"Could not find class {class_name}")

    # Find end of class: next top-level class or top-level decorator followed by class.
    class_end = len(lines)
    for i in range(class_idx + 1, len(lines)):
        line = lines[i]
        if line.startswith("class ") or line.startswith("@OpenAIRequestHandlerFactory.register"):
            class_end = i
            break

    method_idx = None
    needle = f"    def {method_name}("
    for i in range(class_idx + 1, class_end):
        if lines[i].startswith(needle):
            method_idx = i
            break

    if method_idx is None:
        raise RuntimeError(f"Could not find {class_name}.{method_name}")

    start, end = find_indented_block(lines, method_idx, "    ")
    replacement = [line + "\n" for line in new_method.rstrip("\n").splitlines()]
    new_lines = lines[:start] + replacement + lines[end:]
    return "".join(new_lines), True


def replace_top_level_method_in_classlike_file(
    text: str,
    method_name: str,
    new_method: str,
) -> tuple[str, bool]:
    """Replace a method by name without requiring class exact formatting.

    Used for OpenAIHTTPBackend._aiter_lines().
    """
    lines = text.splitlines(keepends=True)
    method_idx = None
    needle = f"    async def {method_name}("
    for i, line in enumerate(lines):
        if line.startswith(needle):
            method_idx = i
            break

    if method_idx is None:
        raise RuntimeError(f"Could not find async method {method_name}")

    start, end = find_indented_block(lines, method_idx, "    ")
    replacement = [line + "\n" for line in new_method.rstrip("\n").splitlines()]
    new_lines = lines[:start] + replacement + lines[end:]
    return "".join(new_lines), True


NEW_AITER_LINES = '''    async def _aiter_lines(self, stream: httpx.Response) -> AsyncIterator[str]:
        """Iterate over complete Server-Sent Event data lines.

        httpx.Response.aiter_lines() splits on every newline. For GPT-OSS text
        completions, the generated text may contain newline characters in the
        streamed JSON payload, which can cause GuideLLM to parse a truncated
        JSON fragment. SSE events are separated by a blank line, so we buffer
        chunks and split by event boundary instead.
        """
        buffer = ""

        async for chunk in stream.aiter_text():
            if not chunk:
                continue

            buffer += chunk

            # Normalize CRLF just in case a server uses Windows-style newlines.
            buffer = buffer.replace("\r\n", "\n")

            while "\n\n" in buffer:
                event, buffer = buffer.split("\n\n", 1)

                data_lines: list[str] = []
                for raw_line in event.split("\n"):
                    line = raw_line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[len("data:") :].strip())

                if not data_lines:
                    continue

                payload = "\n".join(data_lines)
                if payload == "[DONE]":
                    yield "data: [DONE]"
                else:
                    yield f"data: {payload}"

        # Best-effort handling for a final event without a trailing blank line.
        if buffer.strip():
            data_lines = []
            for raw_line in buffer.replace("\r\n", "\n").split("\n"):
                line = raw_line.strip()
                if line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())

            if data_lines:
                payload = "\n".join(data_lines)
                yield "data: [DONE]" if payload == "[DONE]" else f"data: {payload}"'''


NEW_EXTRACT_LINE_DATA = '''    def extract_line_data(self, line: str) -> dict[str, Any] | None:
        """Extract JSON data from a streaming response line.

        Returns:
            None for stream completion, {} for ignored lines, parsed JSON otherwise.

        Raises:
            RuntimeError if the backend sends an error object inside an SSE event.
        """
        if line == "data: [DONE]":
            return None

        if not line or not (line := line.strip()) or not line.startswith("data:"):
            return {}

        payload = line[len("data:") :].strip()
        data = json.loads(payload)

        if isinstance(data, dict) and "error" in data:
            error = data["error"]
            if isinstance(error, dict):
                message = error.get("message") or str(error)
            else:
                message = str(error)
            raise RuntimeError(f"Streaming backend error: {message}")

        return data'''


def patch_http(root: Path, dry_run: bool) -> None:
    path = root / "backends" / "openai" / "http.py"
    if not path.exists():
        raise RuntimeError(f"File not found: {path}")

    text = path.read_text()
    if "SSE events are separated by a blank line" in text:
        print(f"[skip] streaming patch already applied in {path}")
        return

    new_text, changed = replace_top_level_method_in_classlike_file(
        text, "_aiter_lines", NEW_AITER_LINES
    )
    if changed:
        print(f"[patch] replacing OpenAIHTTPBackend._aiter_lines in {path}")
        if not dry_run:
            backup_file(path)
            path.write_text(new_text)


def patch_request_handlers(root: Path, dry_run: bool) -> None:
    path = root / "backends" / "openai" / "request_handlers.py"
    if not path.exists():
        raise RuntimeError(f"File not found: {path}")

    text = path.read_text()
    if "Streaming backend error:" in text and "Returns:" in text:
        print(f"[skip] extract_line_data patch already applied in {path}")
        return

    new_text, changed = replace_method_in_class(
        text,
        "TextCompletionsRequestHandler",
        "extract_line_data",
        NEW_EXTRACT_LINE_DATA,
    )
    if changed:
        print(f"[patch] replacing TextCompletionsRequestHandler.extract_line_data in {path}")
        if not dry_run:
            backup_file(path)
            path.write_text(new_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollback", action="store_true", help="restore backups")
    parser.add_argument("--dry-run", action="store_true", help="show what would change")
    args = parser.parse_args()

    root = find_guidellm_root()
    print(f"[info] guidellm root: {root}")

    files = [
        root / "backends" / "openai" / "http.py",
        root / "backends" / "openai" / "request_handlers.py",
    ]

    if args.rollback:
        for path in files:
            rollback_file(path)
        return 0

    patch_http(root, args.dry_run)
    patch_request_handlers(root, args.dry_run)

    if args.dry_run:
        print("[done] dry run completed; no files changed")
    else:
        print("[done] patched GuideLLM streaming/SSE handling")
        print("[next] restart the process that imports/runs GuideLLM")
        print("[test] python -m py_compile " + " ".join(str(p) for p in files))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
