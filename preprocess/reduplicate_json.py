#!/usr/bin/env python3
"""
llist=(
    graph-sp-dijkstra
    string-manacher
    graph-sp-bellmanford
    string-kmp
    graph-mst-prim
    array-moore
    graph-sp-floyd
    math-euclidean
    array-gray
    math-strassen
)
for cate in ${llist[@]}; do
    CATEGORY=$cate
    LEVEL_LIST=(0 1 2)
    ID_LIST=(0 1 2 3 4 5 6 7)
    cd ${PROJECT_ROOT}
    for level in "${LEVEL_LIST[@]}"; do
        for id in "${ID_LIST[@]}"; do
            echo "Processing Level: $level, ID: $id..."
            python preprocess/reduplicate_json.py \
                --input_path "datasets/final_test/$CATEGORY/level${level}/${id}.json" \
                --output_dir "_data/final_test/$CATEGORY/level${level}" \
                --count 16 \
                --prefix "id${id}" \
                --question_key "problem" \
                --solution_key "" \
                --extra_key "test_cases"
        done
    done
done

"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# @codex: Keep CLI output English-only without changing behavior.

# ANSI color escape sequences.
RED = "\033[91m"
RESET = "\033[0m"

def print_error(message: str):
    """Print an error message in red."""
    print(f"{RED}Error: {message}{RESET}")

def print_warning(message: str):
    """Print a warning message in red."""
    print(f"{RED}Warning: {message}{RESET}")

def load_single_json(input_path: str) -> Dict[str, Any]:
    """Load a single JSON file."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print_error(f"unable to read source file {input_path}: {e}")
        exit(1)


def try_parse_json_string(value: Any) -> Any:
    """Try to parse a JSON-encoded string into a Python object."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def extract_and_map_fields(
    item: Dict[str, Any],
    question_key: str,
    solution_key: Optional[str],
    extra_keys: List[str]
) -> Dict[str, Any]:
    """
    Core transformation logic that maps source keys to standardized keys.
    """
    result = {}

    # 1. Map the question field to 'question'.
    if question_key in item:
        result['question'] = item[question_key]
    else:
        print_warning(f"question field '{question_key}' was not found in the source file")

    # 2. Map the solution field to 'solution' if requested.
    if solution_key:
        if solution_key in item:
            raw_solution = item[solution_key]
            result['solution'] = try_parse_json_string(raw_solution)
        else:
            print_warning(f"solution field '{solution_key}' was not found in the source file")

    # 3. Copy additional requested fields.
    for key in extra_keys:
        if key in item:
            result[key] = try_parse_json_string(item[key])
        else:
            print_warning(f"extra field '{key}' was not found in the source file")

    return result


def main():
    parser = argparse.ArgumentParser(description="Single-JSON copy and field-mapping tool")

    PROJECT_ROOT = Path(__file__).parent.absolute()

    parser.add_argument("--input_path", type=str, required=True, help="Source JSON file path")
    parser.add_argument("--count", type=int, required=True, help="Number of copies to generate")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "_data" / "items"), help="Output directory")
    parser.add_argument("--prefix", type=str, default="problem", help="Output filename prefix")
    parser.add_argument("--question_key", type=str, default="question", help="Key used for the question in the source file")
    parser.add_argument("--solution_key", type=str, default="solution", help="Key used for the solution in the source file")
    parser.add_argument("--extra_key", action="append", default=[], help="Extra key to retain")

    args = parser.parse_args()

    # 1. Load source data.
    raw_data = load_single_json(args.input_path)

    # 2. Transform the data format.
    final_item = extract_and_map_fields(
        raw_data,
        args.question_key,
        args.solution_key,
        args.extra_key
    )

    if not final_item:
        print_error("converted data is empty; check whether the provided keys are correct.")
        return

    # 3. Write output files in batch.
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.count):
        filename = f"{args.prefix}_{i}.json"
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_item, f, indent=2, ensure_ascii=False)

    # 4. Print the final summary.
    print("\n" + "=" * 60)
    print("Generation complete (Summary)")
    print("-" * 60)
    print(f"Source file    : {args.input_path}")
    print(f"Field mapping  : {args.question_key} -> question")
    if args.solution_key:
        print(f"                 {args.solution_key} -> solution")
    print(f"Generated      : {args.count} file(s)")
    print(f"Output dir     : {os.path.abspath(args.output_dir)}")
    print(f"Filename form  : {args.prefix}_[0-{args.count-1}].json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
