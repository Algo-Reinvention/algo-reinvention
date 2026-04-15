from __future__ import annotations

import ast
import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for path in (current, *current.parents):
        if (path / ".git").exists() or (path / ".env.example").exists():
            return path
    raise FileNotFoundError(f"Could not locate repository root from {current}")


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if value:
        in_single = False
        in_double = False
        trimmed_chars: list[str] = []
        for ch in value:
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "#" and not in_single and not in_double:
                break
            trimmed_chars.append(ch)
        value = "".join(trimmed_chars).rstrip()
    if not value:
        return ""
    if value[0] == value[-1] and value[0] in {'"', "'"}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
        return str(parsed)
    return value


def load_repo_env(start: Path | None = None, override: bool = False) -> Path:
    repo_root = find_repo_root(start)
    env_path = repo_root / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            if not override and key in os.environ:
                continue
            os.environ[key] = _parse_env_value(raw_value)
    return repo_root


def repo_env_path(start: Path | None = None) -> Path:
    return find_repo_root(start) / ".env"


def repo_env_has_key(name: str, start: Path | None = None) -> bool:
    env_path = repo_env_path(start)
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _ = line.split("=", 1)
        if key.strip() == name:
            return True
    return False


def require_repo_env_key(name: str, start: Path | None = None) -> str:
    env_path = repo_env_path(start)
    if not env_path.exists():
        raise RuntimeError(
            f"{env_path} does not exist. Copy .env.example to .env and set '{name}' before running this entrypoint."
        )
    if not repo_env_has_key(name, start):
        raise RuntimeError(
            f"'{name}' is not defined in {env_path}. Please set it there before running this entrypoint."
        )
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"'{name}' is empty. Please set it in {env_path}.")
    return value


def require_env_var(name: str, start: Path | None = None) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value

    env_path = repo_env_path(start)
    raise RuntimeError(f"Environment variable '{name}' is not set. Please set it in {env_path} or export it explicitly.")


def expand_env_vars(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value
