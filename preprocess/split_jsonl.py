#!/usr/bin/env python3
"""
General dataset splitting utility.
Supports arbitrary .jsonl datasets.
Splits each item into a separate compact JSON file while keeping only required fields.

cd ${PROJECT_ROOT}
python preprocess/split_jsonl.py \
	--input_path _data/unlearn_test/dijkstra_level0/dijkstra_level0.jsonl \
	--output_dir _data/unlearn_test/dijkstra_level0/split \
	--prefix level0 \
	--question_key problem \
	--solution_key "" \
	--extra_key input_path \
	--extra_key groundtruth_path

"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# @codex: Keep CLI output English-only without changing behavior.

def load_jsonl_data(input_path: str) -> List[Dict[str, Any]]:
    """Load a .jsonl file where each line is a JSON object."""
    print(f"Loading data from {input_path}...")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse JSON on line {line_num}; skipped: {e}")
                continue
    print(f"Loaded {len(data)} record(s) successfully")
    return data


def try_parse_json_string(value: Any) -> Any:
    """Try to parse a JSON-encoded string value into a Python object."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass  # If it is not valid JSON, return the original string unchanged.
    return value


def extract_fields(
    item: Dict[str, Any],
    question_key: str,
    solution_key: Optional[str],
    extra_keys: List[str]
) -> Dict[str, Any]:
    """Extract requested fields from the source item and parse nested JSON strings."""
    result = {}

    # Extract the question text.
    if question_key in item:
        result['question'] = item[question_key]
    else:
        print(f"Warning: missing question field '{question_key}'; it will be omitted")

    # Extract the reference solution if requested.
    if solution_key and solution_key in item:
        raw_solution = item[solution_key]
        parsed = try_parse_json_string(raw_solution)
        result['solution'] = parsed
    elif solution_key:
        print(f"Warning: missing solution field '{solution_key}'; it will be omitted")

    # Extract any additional requested fields.
    for key in extra_keys:
        if key in item:
            raw_val = item[key]
            parsed_val = try_parse_json_string(raw_val)
            result[key] = parsed_val
        else:
            print(f"Warning: missing extra field '{key}'; it will be omitted")

    return result


def save_items_to_files(
    data: List[Dict[str, Any]],
    output_dir: str,
    prefix: str,
    question_key: str,
    solution_key: Optional[str],
    extra_keys: List[str]
):
    """Save each processed item as {prefix}_{id}.json."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving files to directory: {output_dir}")

    saved_count = 0
    for i, item in enumerate(data):
        filtered_item = extract_fields(item, question_key, solution_key, extra_keys)

        # Skip the item if no fields remained after extraction.
        if not filtered_item:
            print(f"Skipping item {i}: no fields remained after extraction")
            continue

        filename = f"{prefix}_{i}.json"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_item, f, indent=2, ensure_ascii=False)

        saved_count += 1

    print(f"Saved {saved_count} compact item file(s) successfully (pattern: {prefix}_{{id}}.json)")


def main():
    parser = argparse.ArgumentParser(
        description="General competitive-programming dataset splitter: split a .jsonl file into compact per-item JSON files"
    )

    # Project root used to build the default output path.
    PROJECT_ROOT = Path(__file__).parent.absolute()

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input .jsonl file path (one JSON object per line)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "_data" / "items"),
        help="Output directory (default: ./_data/items)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="problem",
        help="Output filename prefix, for example 'apps', 'lc', or 'cf' (default: problem)"
    )
    parser.add_argument(
        "--question_key",
        type=str,
        default="question",
        help="Key for the question text (default: question)"
    )
    parser.add_argument(
        "--solution_key",
        type=str,
        default="solution",
        help="Key for the reference solution (default: solution); leave empty if unavailable"
    )
    parser.add_argument(
        "--extra_key",
        action="append",
        default=[],
        help="Extra field name to keep; may be repeated, for example --extra_key input_output --extra_key difficulty"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("General Competitive-Programming Dataset Splitter")
    print("=" * 60)
    print(f"Input file     : {args.input_path}")
    print(f"Output dir     : {args.output_dir}")
    print(f"File prefix    : {args.prefix}")
    print(f"Question key   : {args.question_key}")
    print(f"Solution key   : {args.solution_key if args.solution_key else '(none)'}")
    print(f"Extra keys     : {args.extra_key if args.extra_key else '(none)'}")
    print("-" * 60)

    # Load data.
    data = load_jsonl_data(args.input_path)

    if not data:
        print("Error: no data was loaded; exiting.")
        return

    # Save results.
    save_items_to_files(
        data=data,
        output_dir=args.output_dir,
        prefix=args.prefix,
        question_key=args.question_key,
        solution_key=args.solution_key if args.solution_key else None,
        extra_keys=args.extra_key
    )

    print("\nData split complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
