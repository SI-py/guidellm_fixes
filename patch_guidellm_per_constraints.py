#!/usr/bin/env python3
"""
Patch GuideLLM to support per-strategy constraints in JSON config.

After patch, you can use JSON like:

{
  "profile": "concurrent",
  "rate": [1, 50],
  "max_requests": null,
  "per_constraints": {
    "max_requests": [20, 200]
  }
}

Meaning:
  strategy 0: concurrency/streams = 1,  max_requests = 20
  strategy 1: concurrency/streams = 50, max_requests = 200

Usage inside the same Python environment/container where GuideLLM is installed:
    python patch_guidellm_per_constraints.py

Rollback:
    python patch_guidellm_per_constraints.py --rollback
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


BACKUP_SUFFIX = ".bak_per_constraints"


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


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"Patch already applied for: {label}")
        return text
    if old not in text:
        raise RuntimeError(f"Could not find target block for: {label}")
    return text.replace(old, new, 1)


def patch_schema_entrypoints(root: Path) -> None:
    path = root / "benchmark" / "schemas" / "generative" / "entrypoints.py"
    if not path.exists():
        raise RuntimeError("Could not find benchmark/schemas/generative/entrypoints.py")

    backup_file(path)
    text = path.read_text()

    # Ensure Any is imported if the file already has a typing import.
    if "from typing import" in text:
        import_line = next((line for line in text.splitlines() if line.startswith("from typing import")), None)
        if import_line and "Any" not in import_line:
            text = text.replace(import_line, import_line + ", Any", 1)

    old = '''    max_requests: int | None = Field(
        default=None, description="Maximum number of requests to execute"
    )'''
    new = '''    max_requests: int | None = Field(
        default=None, description="Maximum number of requests to execute"
    )
    per_constraints: dict[str, list[int | float]] | None = Field(
        default=None,
        description=(
            "Per-strategy constraints. Example: "
            "{'max_requests': [20, 200]} for multi-stage profiles."
        ),
    )'''
    text = replace_once(text, old, new, "schema per_constraints field")

    validator = '''
    @field_validator("per_constraints", mode="before")
    @classmethod
    def validate_per_constraints(cls, value: Any) -> dict[str, list[int | float]] | None:
        if value is None:
            return None

        if not isinstance(value, dict):
            raise ValueError("per_constraints must be a dictionary")

        allowed_keys = {
            "max_seconds",
            "max_requests",
            "max_errors",
            "max_error_rate",
            "max_global_error_rate",
        }

        for key, values in value.items():
            if key not in allowed_keys:
                raise ValueError(
                    f"Unsupported per_constraints key '{key}'. "
                    f"Allowed keys: {sorted(allowed_keys)}"
                )

            if not isinstance(values, list):
                raise ValueError(
                    f"per_constraints['{key}'] must be a list, "
                    f"got {type(values).__name__}"
                )

            if not values:
                raise ValueError(f"per_constraints['{key}'] must not be empty")

            if any(v is None for v in values):
                raise ValueError(f"per_constraints['{key}'] must not contain null values")

        return value

'''

    if "def validate_per_constraints" not in text:
        if "    model_config =" in text:
            text = text.replace("    model_config =", validator + "    model_config =", 1)
        else:
            # Fallback: append inside class before EOF is risky, so fail loudly.
            raise RuntimeError("Could not find model_config to insert per_constraints validator")
    else:
        print("Schema validator already exists")

    path.write_text(text)
    print(f"Patched: {path}")


def patch_entrypoints(root: Path) -> None:
    path = root / "benchmark" / "entrypoints.py"
    if not path.exists():
        raise RuntimeError("Could not find benchmark/entrypoints.py")

    backup_file(path)
    text = path.read_text()

    old = '''    over_saturation: dict[str, Any] | None = None,
    console: Console | None = None,'''
    new = '''    over_saturation: dict[str, Any] | None = None,
    per_constraints: dict[str, list[int | float]] | None = None,
    console: Console | None = None,'''
    text = replace_once(text, old, new, "resolve_profile per_constraints parameter")

    old = '''    if not isinstance(profile, Profile):
        profile = ProfileFactory.create(
            profile,
            random_seed,
            constraints,
            **profile_kwargs,
        )'''
    new = '''    if not isinstance(profile, Profile):
        if per_constraints:
            profile_kwargs["per_constraints"] = per_constraints

        profile = ProfileFactory.create(
            profile,
            random_seed,
            constraints,
            **profile_kwargs,
        )

        if hasattr(profile, "validate_per_constraints_length"):
            profile.validate_per_constraints_length()'''
    text = replace_once(text, old, new, "pass per_constraints to ProfileFactory")

    old = '''        over_saturation=args.over_saturation,
        console=console,'''
    new = '''        over_saturation=args.over_saturation,
        per_constraints=args.per_constraints,
        console=console,'''
    text = replace_once(text, old, new, "pass args.per_constraints to resolve_profile")

    path.write_text(text)
    print(f"Patched: {path}")


def patch_profiles(root: Path) -> None:
    path = root / "benchmark" / "profiles.py"
    if not path.exists():
        raise RuntimeError("Could not find benchmark/profiles.py")

    backup_file(path)
    text = path.read_text()

    old = '''    constraints: dict[
        str,
        Any | dict[str, Any] | ConstraintInitializer,
    ] | None = Field(
        default=None,
        description="Constraints to apply to the strategy",
    )'''
    new = '''    constraints: dict[
        str,
        Any | dict[str, Any] | ConstraintInitializer,
    ] | None = Field(
        default=None,
        description="Constraints to apply to the strategy",
    )
    per_constraints: dict[str, list[Any]] | None = Field(
        default=None,
        description="Per-strategy runtime constraints applied by strategy index",
    )'''
    text = replace_once(text, old, new, "Profile per_constraints field")

    old = '''    def next_strategy_constraints(
        self,
        next_strategy: SchedulingStrategy | None,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> dict[str, Constraint] | None:
        _ = (prev_strategy, prev_benchmark)

        return (
            ConstraintsInitializerFactory.resolve(self.constraints)
            if next_strategy and self.constraints
            else None
        )'''
    new = '''    def next_strategy_constraints(
        self,
        next_strategy: SchedulingStrategy | None,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> dict[str, Constraint] | None:
        _ = (prev_strategy, prev_benchmark)

        if not next_strategy:
            return None

        strategy_index = len(self.completed_strategies)
        final_constraints: dict[str, Any] = dict(self.constraints or {})

        if self.per_constraints:
            for key, values in self.per_constraints.items():
                if strategy_index >= len(values):
                    raise ValueError(
                        f"per_constraints['{key}'] has {len(values)} values, "
                        f"but strategy index {strategy_index} is being requested. "
                        "The list length must match the number of profile strategies."
                    )
                final_constraints[key] = values[strategy_index]

        return (
            ConstraintsInitializerFactory.resolve(final_constraints)
            if final_constraints
            else None
        )'''
    text = replace_once(text, old, new, "Profile next_strategy_constraints")

    validator = '''
    def validate_per_constraints_length(self) -> None:
        if not self.per_constraints:
            return

        expected = len(self.strategy_types)

        for key, values in self.per_constraints.items():
            if len(values) != expected:
                raise ValueError(
                    f"per_constraints['{key}'] length must match number of strategies. "
                    f"Got {len(values)} values for {expected} strategies."
                )

'''
    if "def validate_per_constraints_length" not in text:
        marker = "    @property\n    def strategy_types"
        if marker not in text:
            raise RuntimeError("Could not find strategy_types property to insert validator")
        text = text.replace(marker, validator + marker, 1)
    else:
        print("Profile per_constraints length validator already exists")

    path.write_text(text)
    print(f"Patched: {path}")


def patch() -> None:
    root = find_guidellm_root()
    print(f"Found GuideLLM root: {root}")
    patch_schema_entrypoints(root)
    patch_entrypoints(root)
    patch_profiles(root)
    print("\nDone. Restart the GuideLLM process after patching.")


def rollback() -> None:
    root = find_guidellm_root()
    print(f"Found GuideLLM root: {root}")
    for rel in [
        "benchmark/schemas/generative/entrypoints.py",
        "benchmark/entrypoints.py",
        "benchmark/profiles.py",
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
