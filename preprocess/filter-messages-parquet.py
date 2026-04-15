#!/usr/bin/env python3
# @codex

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a messages parquet by dropping rows whose rendered chat text contains target keywords."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input parquet file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the filtered parquet file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the HF tokenizer.")
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=[],
        help="Keywords to drop, matched case-insensitively against rendered chat text.",
    )
    return parser.parse_args()


def should_drop_row(messages, tokenizer, keywords_lower: list[str]) -> bool:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    rendered_lower = rendered.lower()
    return any(keyword in rendered_lower for keyword in keywords_lower)


def main() -> int:
    args = parse_args()
    keywords_lower = [keyword.lower() for keyword in args.keywords]

    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    print(f"Reading input parquet: {args.input_path}")
    dataframe = pd.read_parquet(args.input_path)
    if "messages" not in dataframe.columns:
        raise ValueError("Input parquet must contain a 'messages' column.")

    kept_rows = []
    dropped_rows = 0
    for _, row in dataframe.iterrows():
        row_dict = row.to_dict()
        if should_drop_row(row_dict["messages"], tokenizer, keywords_lower):
            dropped_rows += 1
            continue
        kept_rows.append(row_dict)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered = pd.DataFrame(kept_rows, columns=dataframe.columns)
    filtered.to_parquet(output_path, index=False)

    print(f"Total rows: {len(dataframe)}")
    print(f"Dropped rows: {dropped_rows}")
    print(f"Kept rows: {len(filtered)}")
    print(f"Saved filtered parquet to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
