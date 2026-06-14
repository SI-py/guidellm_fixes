#!/usr/bin/env python3
"""
Patch GuideLLM 0.6.x OpenAI HTTP streaming/SSE handling.
Fixed version: generated code uses escaped backslashes correctly.

Usage:
  python patch_guidellm_streaming_sse_v060_fixed.py --repair-and-patch
  python patch_guidellm_streaming_sse_v060_fixed.py --rollback
  python patch_guidellm_streaming_sse_v060_fixed.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BACKUP_SUFFIXES = [
    ".bak_streaming_sse_v060",
    ".bak_streaming_sse",
]
NEW_BACKUP_SUFFIX = ".bak_streaming_sse_v060_fixed"


def find_guidellm_root() -> Path:
    try:
        import guidellm  # type: ignore
    except ImportError as exc:
        raise RuntimeError("guidellm is not installed in this Python environment") from exc
    return Path(guidellm.__file__).resolve().parent


def backup_file(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + NEW_BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[backup] {path} -> {backup}")
    else:
        print(f"[backup] already exists: {backup}")
    return backup


def restore_any_backup(path: Path) -> bool:
    for suffix in [NEW_BACKUP_SUFFIX] + BACKUP_SUFFIXES:
        backup = path.with_suffix(path.suffix + suffix)
        if backup.exists():
            shutil.copy2(backup, path)
            print(f"[restore] {path} <- {backup}")
            return True
    print(f"[restore] no backup found for {path}")
    return False


def find_indented_block(lines: list[str], start_idx: int, indent: str) -> tuple[int, int]:
    base_len = len(indent)
    end = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip(" "))
        if leading <= base_len:
            end = i
            break
    return start_idx, end


def replace_async_method(text: str, method_name: str, new_method: str) -> str:
    lines = text.splitlines(keepends=True)
    needle = f"    async def {method_name}("
    method_idx = None
    for i, line in enumerate(lines):
        if line.startswith(needle):
            method_idx = i
            break
    if method_idx is None:
        raise RuntimeError(f"Could not find async method {method_name}")
    start, end = find_indented_block(lines, method_idx, "    ")
    replacement = [line + "\n" for line in new_method.rstrip("\n").splitlines()]
    return "".join(lines[:start] + replacement + lines[end:])


def replace_method_in_class(text: str, class_name: str, method_name: str, new_method: str) -> str:
    lines = text.splitlines(keepends=True)
    class_idx = None
    for i, line in enumerate(lines):
        if line.startswith(f"class {class_name}"):
            class_idx = i
            break
    if class_idx is None:
        raise RuntimeError(f"Could not find class {class_name}")

    class_end = len(lines)
    for i in range(class_idx + 1, len(lines)):
        line = lines[i]
        if line.startswith("class ") or line.startswith("@OpenAIRequestHandlerFactory.register"):
            class_end = i
            break

    needle = f"    def {method_name}("
    method_idx = None
    for i in range(class_idx + 1, class_end):
        if lines[i].startswith(needle):
            method_idx = i
            break
    if method_idx is None:
        raise RuntimeError(f"Could not find {class_name}.{method_name}")

    start, end = find_indented_block(lines, method_idx, "    ")
    replacement = [line + "\n" for line in new_method.rstrip("\n").splitlines()]
    return "".join(lines[:start] + replacement + lines[end:])


NEW_AITER_LINES = r'''    async def _aiter_lines(self, stream: httpx.Response) -> AsyncIterator[str]:
        """Iterate over complete Server-Sent Event data lines.

        httpx.Response.aiter_lines() splits on every newline. For GPT-OSS text
        completions, generated text may contain newline characters in the
        streamed JSON payload. SSE events are separated by a blank line, so we
        buffer chunks and split by event boundary instead.
        """
        buffer = ""

        async for chunk in stream.aiter_text():
            if not chunk:
                continue

            buffer += chunk
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
            data_lines: list[str] = []
            for raw_line in buffer.replace("\r\n", "\n").split("\n"):
                line = raw_line.strip()
                if line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())

            if data_lines:
                payload = "\n".join(data_lines)
                yield "data: [DONE]" if payload == "[DONE]" else f"data: {payload}"'''


NEW_EXTRACT_LINE_DATA = r'''    def extract_line_data(self, line: str) -> dict[str, Any] | None:
        """Extract JSON data from a streaming response line."""
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
    new_text = replace_async_method(text, "_aiter_lines", NEW_AITER_LINES)
    print(f"[patch] OpenAIHTTPBackend._aiter_lines in {path}")
    if not dry_run:
        backup_file(path)
        path.write_text(new_text)


def patch_request_handlers(root: Path, dry_run: bool) -> None:
    path = root / "backends" / "openai" / "request_handlers.py"
    if not path.exists():
        raise RuntimeError(f"File not found: {path}")
    text = path.read_text()
    new_text = replace_method_in_class(
        text,
        "TextCompletionsRequestHandler",
        "extract_line_data",
        NEW_EXTRACT_LINE_DATA,
    )
    print(f"[patch] TextCompletionsRequestHandler.extract_line_data in {path}")
    if not dry_run:
        backup_file(path)
        path.write_text(new_text)


def compile_check(files: list[Path]) -> None:
    import py_compile
    for path in files:
        py_compile.compile(str(path), doraise=True)
        print(f"[compile] OK {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--repair-and-patch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = find_guidellm_root()
    print(f"[info] guidellm root: {root}")
    files = [
        root / "backends" / "openai" / "http.py",
        root / "backends" / "openai" / "request_handlers.py",
    ]

    if args.rollback:
        for path in files:
            restore_any_backup(path)
        compile_check([p for p in files if p.exists()])
        return 0

    if args.repair_and_patch:
        for path in files:
            restore_any_backup(path)

    patch_http(root, args.dry_run)
    patch_request_handlers(root, args.dry_run)

    if not args.dry_run:
        compile_check(files)
        print("[done] patched. Restart GuideLLM process.")
    else:
        print("[done] dry run completed; no files changed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
