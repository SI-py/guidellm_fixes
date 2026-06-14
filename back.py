from pathlib import Path
import guidellm

root = Path(guidellm.__file__).resolve().parent
print(f"GuideLLM root: {root}")

files = [
    "backends/openai/http.py",
    "backends/openai/request_handlers.py",
    "benchmark/schemas/generative/entrypoints.py",
    "benchmark/entrypoints.py",
    "benchmark/profiles.py",
]

for rel in files:
    path = root / rel

    possible_backups = [
        path.with_suffix(path.suffix + ".bak_streaming_sse"),
        path.with_suffix(path.suffix + ".bak_per_constraints"),
        path.with_suffix(path.suffix + ".bak"),
    ]

    restored = False

    for backup in possible_backups:
        if backup.exists():
            path.write_text(backup.read_text())
            print(f"Restored: {path} from {backup}")
            restored = True
            break

    if not restored:
        print(f"No backup found for: {path}")