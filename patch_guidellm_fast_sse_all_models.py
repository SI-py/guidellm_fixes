#!/usr/bin/env python3
"""
Patch GuideLLM 0.6.0 with a fast tolerant SSE line parser for all models.

What changes:
- Keeps stream.aiter_lines() as the transport path.
- Valid JSON SSE data lines are yielded immediately.
- Buffering is used ONLY after a JSONDecodeError, to recover split/truncated JSON.
- Handles data: {"error": ...} as RuntimeError.
- Adds hard limits to avoid infinite pending/hanging.

Usage:
    python patch_guidellm_fast_sse_all_models.py

Rollback:
    python patch_guidellm_fast_sse_all_models.py --rollback

Optional limits:
    GUIDELLM_SSE_MAX_BUFFER=1048576
    GUIDELLM_SSE_MAX_LINES=256
"""

from __future__ import annotations

import argparse
import py_compile
import re
import shutil
import sys
from pathlib import Path

BACKUP_SUFFIX = ".bak_fast_sse_all_models"


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
        print(f"[backup] {backup}")
    else:
        print(f"[backup exists] {backup}")


def rollback_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        print(f"[rollback] no backup found for {path}")
        return
    shutil.copy2(backup, path)
    print(f"[rollback] restored {path}")


def find_async_function_block(text: str, func_name: str) -> tuple[int, int, str]:
    pattern = re.compile(rf"(?m)^([ \t]*)async def {re.escape(func_name)}\s*\(")
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"Could not find async function '{func_name}'")

    start = match.start()
    indent = match.group(1)
    lines = text[start:].splitlines(keepends=True)
    consumed = len(lines[0])
    end = len(text)

    for line in lines[1:]:
        stripped = line.lstrip()
        line_indent = line[: len(line) - len(stripped)]
        if stripped and not stripped.startswith("#"):
            if len(line_indent) <= len(indent) and (
                stripped.startswith("def ")
                or stripped.startswith("async def ")
                or stripped.startswith("class ")
                or stripped.startswith("@")
            ):
                end = start + consumed
                break
        consumed += len(line)

    return start, end, indent


def ensure_imports(text: str) -> str:
    if "import json" not in text:
        text = "import json\n" + text
    if "import os" not in text:
        text = "import os\n" + text

    if "import orjson" not in text and "orjson = None" not in text:
        import_matches = list(re.finditer(r"(?m)^(import .+|from .+ import .+)\n", text))
        insert_at = import_matches[-1].end() if import_matches else 0
        block = (
            "\ntry:\n"
            "    import orjson  # type: ignore\n"
            "except ImportError:  # pragma: no cover\n"
            "    orjson = None  # type: ignore\n"
        )
        text = text[:insert_at] + block + text[insert_at:]

    return text


FAST_AITER_LINES_TEMPLATE = """{indent}async def _aiter_lines(self, stream):
{indent}    # Fast tolerant SSE parser.
{indent}    # Normal valid SSE data lines are yielded immediately.
{indent}    # Recovery buffering is used only after JSONDecodeError.
{indent}
{indent}    def _loads_json(payload: str):
{indent}        if orjson is not None:
{indent}            return orjson.loads(payload)
{indent}        return json.loads(payload)
{indent}
{indent}    def _is_json_decode_error(exc: Exception) -> bool:
{indent}        if isinstance(exc, json.JSONDecodeError):
{indent}            return True
{indent}        if orjson is not None and isinstance(exc, orjson.JSONDecodeError):
{indent}            return True
{indent}        return False
{indent}
{indent}    pending_payload = None
{indent}    pending_lines = 0
{indent}    max_pending_bytes = int(os.getenv("GUIDELLM_SSE_MAX_BUFFER", "1048576"))
{indent}    max_pending_lines = int(os.getenv("GUIDELLM_SSE_MAX_LINES", "256"))
{indent}
{indent}    async for raw_line in stream.aiter_lines():
{indent}        if not raw_line:
{indent}            continue
{indent}
{indent}        line = raw_line.strip()
{indent}
{indent}        if line == "data: [DONE]":
{indent}            if pending_payload is not None:
{indent}                raise RuntimeError(
{indent}                    "SSE stream finished with incomplete JSON payload: "
{indent}                    f"{{pending_payload[:500]!r}}"
{indent}                )
{indent}            yield line
{indent}            continue
{indent}
{indent}        if pending_payload is None:
{indent}            if not line.startswith("data:"):
{indent}                # Preserve original behavior for non-data lines: skip empty/control lines.
{indent}                continue
{indent}            payload = line[len("data:") :].strip()
{indent}            pending_lines = 1
{indent}        else:
{indent}            if line.startswith("data:"):
{indent}                continuation = line[len("data:") :].strip()
{indent}            else:
{indent}                continuation = raw_line
{indent}            payload = pending_payload + "\\n" + continuation
{indent}            pending_lines += 1
{indent}
{indent}        if len(payload) > max_pending_bytes:
{indent}            raise RuntimeError(
{indent}                "SSE JSON recovery buffer exceeded limit: "
{indent}                f"{{len(payload)}} bytes"
{indent}            )
{indent}
{indent}        if pending_lines > max_pending_lines:
{indent}            raise RuntimeError(
{indent}                "SSE JSON recovery exceeded max line count: "
{indent}                f"{{pending_lines}} lines"
{indent}            )
{indent}
{indent}        try:
{indent}            data = _loads_json(payload)
{indent}        except Exception as exc:
{indent}            if _is_json_decode_error(exc):
{indent}                pending_payload = payload
{indent}                continue
{indent}            raise
{indent}
{indent}        if isinstance(data, dict) and "error" in data:
{indent}            error = data["error"]
{indent}            if isinstance(error, dict):
{indent}                message = error.get("message", str(error))
{indent}            else:
{indent}                message = str(error)
{indent}            raise RuntimeError(f"Streaming backend error: {{message}}")
{indent}
{indent}        yield "data: " + payload
{indent}        pending_payload = None
{indent}        pending_lines = 0
{indent}
{indent}    if pending_payload is not None:
{indent}        raise RuntimeError(
{indent}            "SSE stream ended before incomplete JSON payload was recovered: "
{indent}            f"{{pending_payload[:500]!r}}"
{indent}        )
"""


def patch_http_py(root: Path, dry_run: bool = False) -> None:
    path = root / "backends" / "openai" / "http.py"
    if not path.exists():
        raise RuntimeError(f"Could not find {path}")

    text = path.read_text()

    if "Fast tolerant SSE parser" in text:
        print(f"[skip] fast SSE patch already appears to be applied in {path}")
        return

    start, end, indent = find_async_function_block(text, "_aiter_lines")
    new_func = FAST_AITER_LINES_TEMPLATE.format(indent=indent)
    new_text = text[:start] + new_func + text[end:]
    new_text = ensure_imports(new_text)

    if dry_run:
        print(f"[dry-run] would patch {path}")
        return

    backup_file(path)
    path.write_text(new_text)
    py_compile.compile(str(path), doraise=True)
    print(f"[patched] {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = find_guidellm_root()
    print(f"[guidellm root] {root}")

    http_py = root / "backends" / "openai" / "http.py"

    if args.rollback:
        rollback_file(http_py)
        py_compile.compile(str(http_py), doraise=True)
        print("[compile] OK")
        return

    patch_http_py(root, dry_run=args.dry_run)

    if not args.dry_run:
        print("")
        print("Done. Fast tolerant SSE parser is enabled for all models.")
        print("Rollback:")
        print(f"  python {Path(__file__).name} --rollback")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
