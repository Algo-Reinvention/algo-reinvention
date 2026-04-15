#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


METADATA_DIR = Path(__file__).resolve().parent
QWEN_MODEL_ALIASES = {
    "qwen3-4b-thinking-2507": "qwen3-4b-thinking-2507",
    "qwen3-4b-instruct-2507": "qwen3-4b-instruct-2507",
}


def resolve_path(name: str) -> Path:
    candidates = [name]
    alias = QWEN_MODEL_ALIASES.get(name)
    if alias:
        candidates.append(alias)

    for candidate in candidates:
        path = METADATA_DIR / f"{candidate}.json"
        if path.is_file():
            return path

    known = ", ".join(sorted(path.stem for path in METADATA_DIR.glob("*.json")))
    raise FileNotFoundError(f"Metadata '{name}' not found. Available entries: {known}")


def load_field(name: str, field: str):
    data = json.loads(resolve_path(name).read_text(encoding="utf-8"))
    if field not in data:
        raise KeyError(f"Field '{field}' not found in metadata '{name}'")
    return data[field]


def main() -> int:
    parser = argparse.ArgumentParser(description="Load a metadata field from Algo_test/metadata.")
    parser.add_argument("--name", required=True, help="Metadata file stem, without .json.")
    parser.add_argument("--field", required=True, help="Field name inside the JSON file.")
    parser.add_argument(
        "--format",
        choices=("raw", "lines", "json"),
        default="raw",
        help="Output format. Use 'lines' for string arrays.",
    )
    args = parser.parse_args()

    try:
        value = load_field(args.name, args.field)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.format == "raw":
        if isinstance(value, str):
            sys.stdout.write(value)
            return 0
        print("The 'raw' format only supports string values.", file=sys.stderr)
        return 1

    if args.format == "lines":
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            for item in value:
                print(item)
            return 0
        print("The 'lines' format only supports string arrays.", file=sys.stderr)
        return 1

    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
