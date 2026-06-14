#!/usr/bin/env python3
"""
Patch GuideLLM OpenAI streaming/SSE handling.

What it fixes:
1. Replaces line-based SSE reading via httpx.Response.aiter_lines()
   with event-based parsing by SSE boundary (blank line / \n\n).
2. Adds explicit handling for streaming payloads like:
   data: {"error": {...}}

Usage inside the same Python environment/container where GuideLLM is installed:
    python patch_guidellm_streaming_sse.py

Rollback:
    python patch_guidellm_streaming_sse.py --rollback
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


BACKUP_SUFFIX = ".bak_streaming_sse"


def find_guidellm_root() -> Path:
    try:
        import guidellm  # type: ignore
    except ImportError as exc:
        raise RuntimeError("guidellm is not installed in this Python environment") from exc

    return Path(guidellm.__file__).resolve().parent


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup created: {backup}")
    else:
        print(f"Backup already exists: {backup}")


def rollback_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        print(f"No backup found for {path}")
        return
    shutil.copy2(backup, path)
    print(f"Restored: {path}")


def replace_method_by_name(text: str, method_name: str, new_method: str) -> str:
    marker = f"    async def {method_name}(" if method_name.startswith("_aiter") else f"    def {method_name}("
    start = text.find(marker)
    if start == -1:
        raise RuntimeError(f"Could not find method {method_name}")

    # Find next top-level class method/property at 4-space indentation.
    next_candidates = []
    for token in ["\n    async def ", "\n    def ", "\n    @"]:
        pos = text.find(token, start + 1)
        if pos != -1:
            next_candidates.append(pos)
    end = min(next_candidates) if next_candidates else len(text)

    old_block = text[start:end].rstrip()
    if new_method.strip() in old_block:
        print(f"Method {method_name} already appears patched")
        return text

    return text[:start] + new_method.rstrip() + "\n" + text[end:]


def patch_http_py(root: Path) -> None:
    path = root / "benchmark" / "backends" / "openai" / "http.py"
    if not path.exists():
        path = root / "backends" / "openai" / "http.py"
    if not path.exists():
        raise RuntimeError("Could not find OpenAI http.py in installed GuideLLM")

    backup_file(path)
    text = path.read_text()

    if "stream.aiter_text()" in text and "SSE event boundary" in text:
        print(f"Streaming SSE patch already present in {path}")
        return

    new_method = '''    async def _aiter_lines(self, stream):
        """Yield complete SSE data lines from an HTTP streaming response.

        GuideLLM previously used stream.aiter_lines(), which can split a single
        SSE event whenever the generated text contains a literal newline. For
        GPT-OSS /v1/completions this can produce truncated JSON chunks and then
        JSONDecodeError in request_handlers.extract_line_data().

        SSE events are separated by a blank line, so we buffer text chunks and
        split by the SSE event boundary instead of raw lines.
        """
        buffer = ""

        async for chunk in stream.aiter_text():
            if not chunk:
                continue

            buffer += chunk
            buffer = buffer.replace("\\r\\n", "\\n")

            while "\\n\\n" in buffer:
                event, buffer = buffer.split("\\n\\n", 1)
                data_lines = []

                for raw_line in event.split("\\n"):
                    line = raw_line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[len("data:"):].strip())

                if not data_lines:
                    continue

                payload = "\\n".join(data_lines)
                if payload == "[DONE]":
                    yield "data: [DONE]"
                else:
                    yield f"data: {payload}"

        # Flush a final partial event only if it looks complete enough to parse.
        # This keeps behavior tolerant for servers that omit the trailing blank line.
        tail = buffer.strip()
        if tail:
            data_lines = []
            for raw_line in tail.split("\\n"):
                line = raw_line.strip()
                if line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())

            if data_lines:
                payload = "\\n".join(data_lines)
                if payload == "[DONE]":
                    yield "data: [DONE]"
                else:
                    yield f"data: {payload}"
'''

    text = replace_method_by_name(text, "_aiter_lines", new_method)
    path.write_text(text)
    print(f"Patched: {path}")


def patch_request_handlers_py(root: Path) -> None:
    path = root / "benchmark" / "backends" / "openai" / "request_handlers.py"
    if not path.exists():
        path = root / "backends" / "openai" / "request_handlers.py"
    if not path.exists():
        raise RuntimeError("Could not find OpenAI request_handlers.py in installed GuideLLM")

    backup_file(path)
    text = path.read_text()

    if "Streaming backend error" in text:
        print(f"Streaming error handling patch already present in {path}")
        return

    # Patch all simple json.loads(line) occurrences in extract_line_data-like methods.
    old = """        line = line[len(\"data:\") :].strip()\n        return json.loads(line)"""
    new = """        line = line[len(\"data:\") :].strip()\n        data = json.loads(line)\n\n        if isinstance(data, dict) and \"error\" in data:\n            error = data[\"error\"]\n            if isinstance(error, dict):\n                message = error.get(\"message\", str(error))\n            else:\n                message = str(error)\n            raise RuntimeError(f\"Streaming backend error: {message}\")\n\n        return data"""

    if old not in text:
        raise RuntimeError(
            "Could not find the expected extract_line_data json.loads block. "
            "Your GuideLLM version may differ; patch manually around extract_line_data()."
        )

    text = text.replace(old, new)
    path.write_text(text)
    print(f"Patched: {path}")


def patch() -> None:
    root = find_guidellm_root()
    print(f"Found GuideLLM root: {root}")
    patch_http_py(root)
    patch_request_handlers_py(root)
    print("\nDone. Restart the GuideLLM process after patching.")


def rollback() -> None:
    root = find_guidellm_root()
    print(f"Found GuideLLM root: {root}")
    for rel in [
        "benchmark/backends/openai/http.py",
        "backends/openai/http.py",
        "benchmark/backends/openai/request_handlers.py",
        "backends/openai/request_handlers.py",
    ]:
        path = root / rel
        if path.exists():
            rollback_file(path)
    print("\nRollback finished. Restart the GuideLLM process.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollback", action="store_true", help="restore backup files")
    args = parser.parse_args()

    try:
        if args.rollback:
            rollback()
        else:
            patch()
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
