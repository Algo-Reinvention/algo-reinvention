"""

python ${PROJECT_ROOT}/preprocess/sub_parquet.py \
    --input_path ${PROJECT_ROOT}/_data/unlearn/qwen3_think_2507_dijkstra/0110_code_debug.parquet \
    -n 128

"""

import argparse
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def head_parquet(args):
    input_path = Path(args.input_path)
    n = args.n

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    # 1. Use pyarrow metadata to inspect the total row count.
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    print(f"Total rows in source: {total_rows}")

    # 3. Stop early if the source file contains fewer than n rows.
    if total_rows < n:
        raise ValueError(f"Error: Requested {n} rows, but the file only contains {total_rows} rows.")

    # 2. Build the output path: path/file-n.parquet.
    output_path = input_path.with_name(f"{input_path.stem}-{n}{input_path.suffix}")

    # 4. Read the first n rows.
    print(f"Reading first {n} rows...")
    # Use pandas read_parquet for portability, then keep only the head.
    df = pd.read_parquet(input_path, engine='pyarrow').head(n)

    # 5. Save the result.
    print(f"Saving to: {output_path}")
    df.to_parquet(output_path, index=False)
    print("Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take the first n rows of a parquet file.")
    
    parser.add_argument("--input_path", type=str, required=True, help="Path to the source parquet file.")
    parser.add_argument("-n", type=int, required=True, help="Number of rows to extract.")

    args = parser.parse_args()
    head_parquet(args)
