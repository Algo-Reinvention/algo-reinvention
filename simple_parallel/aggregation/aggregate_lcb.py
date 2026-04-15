#!/usr/bin/env python3
"""
    Aggregate the mean and standard deviation of public, private,
    and total pass rates.

python ${PROJECT_ROOT}/simple_parallel/aggregation/aggregate_lcb.py \
    --input_dir ${PROJECT_ROOT}/_output/results/qwen3-4b-unlearn/think/1224_llm_as_a_judge_idk/seed2/benchmarks/lcbv6-8
"""

import os
import json
import argparse
import numpy as np  # Use NumPy for matrix operations.
from pathlib import Path
from typing import Dict, List, Any

def load_and_extract_matrices(output_dir: str) -> Dict[str, List[List[int]]]:
    """
    Load all evaluation results and extract them directly into lists of
    0/1 matrix rows.
    Returns: {'public': [[1,0,1], ...], 'private': [...], 'total': [...]}
    """
    # Initialize the storage structure.
    matrices = {
        'public': [],
        'private': [],
        'total': []
    }
    
    file_count = 0
    print(f"Loading evaluation results from {output_dir}...")

    # Recursively scan for JSON result files.
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Verify that this is a valid evaluation result file.
                    if 'evaluation_results' not in data or not data['evaluation_results']:
                        continue

                    # Extract one matrix row for the current problem.
                    row_data = {'public': [], 'private': [], 'total': []}
                    
                    for eval_res in data['evaluation_results']:
                        stats = eval_res.get('stats', {})
                        
                        # --- Force binarization ---
                        # Iterate over public, private, and total.
                        for key in ['public', 'private', 'total']:
                            pass_rate = stats.get(key, {}).get('pass_rate', 0.0)
                            # Count only a full pass (1.0) as 1; otherwise 0.
                            val = 1 if pass_rate == 1.0 else 0
                            row_data[key].append(val)
                    
                    # Append the row as long as it contains data.
                    if row_data['total']: # Confirm that data exists.
                        for key in matrices:
                            matrices[key].append(row_data[key])
                        file_count += 1

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Skipping invalid file {file_path}: {e}")
                    continue

    print(f"Successfully loaded evaluation results for {file_count} problems")
    return matrices


def calculate_statistics(matrix_rows: List[List[int]]) -> Dict[str, Any]:
    """
    Run the core NumPy-based aggregation logic (similar to aggregate_math.py).
    """
    if not matrix_rows:
        return {'mean': 0.0, 'std': 0.0, 'count': 0, 'max_k': 0}

    # 1. Determine the largest K (number of responses).
    max_k = max(len(row) for row in matrix_rows)

    # 2. Normalize the matrix by padding every row to length max_k with 0.
    # Missing generations are treated as failures.
    padded_matrix = np.array([row + [0] * (max_k - len(row)) for row in matrix_rows])

    # 3. Core computation.
    # axis=0 computes the column-wise mean -> [Acc@1, Acc@2, ... Acc@K]
    acc_per_k = padded_matrix.mean(axis=0)
    
    # Compute the mean and standard deviation across Acc values.
    final_mean = acc_per_k.mean()
    final_std = acc_per_k.std()

    # Convert Acc@k into a dictionary for readability.
    acc_dict = {f"{k+1}": float(v) for k, v in enumerate(acc_per_k)}

    return {
        'mean': float(final_mean),
        'std': float(final_std),
        'count': len(matrix_rows),     # Total number of problems.
        'max_k': max_k,                # Maximum number of attempts.
        'acc_per_k': acc_dict          # Independent accuracy for each round.
    }


def aggregate_all_metrics(matrices: Dict[str, List[List[int]]]) -> Dict[str, Any]:
    """Compute statistics for all metrics."""
    results = {}
    for key, rows in matrices.items():
        results[key] = calculate_statistics(rows)
    return results


def print_summary(analysis: Dict[str, Any]):
    """Print the aggregation summary."""
    print("\n" + "="*60)
    print("📊 LCB Evaluation Report (NumPy / Binarized / Stability)")
    print("="*60)

    for key in ['public', 'private', 'total']:
        stats = analysis.get(key)
        if not stats: continue
        
        print(f"\n🏷️  [{key.upper()}] Statistics:")
        print(f"  Total problems: {stats['count']}")
        print(f"  Maximum attempts (K): {stats['max_k']}")
        print(f"  Acc@k details (first 5): {dict(list(stats['acc_per_k'].items())[:5])}...")
        print("-" * 30)
        print(f"  ⭐ Mean Acc: {stats['mean']:.4f} ({stats['mean'] * 100:.2f}%)")
        print(f"  📉 Standard deviation: {stats['std']:.4f}")


def save_aggregated_results(analysis: Dict[str, Any], output_file: str):
    """Save aggregated results."""
    # Ensure the output directory exists.
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metadata': {
            'timestamp': __import__('time').time(),
            'method': 'numpy_matrix_binarized_stability'
        },
        'metrics': analysis
    }
    output_data['mean'] = analysis['total']['mean']
    output_data['std'] = analysis['total']['std']

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Aggregated results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")

    # Resolve the project root.
    PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

    parser.add_argument("--input_dir", type=str,
                       default=str(PROJECT_ROOT / "_output"),
                       help="Evaluation results directory")
    parser.add_argument("--output_file", type=str,
                       help="Output file path")

    args = parser.parse_args()

    # Select a default output location automatically.
    if not args.output_file:
        aggregation_dir = os.path.join(args.input_dir, "aggregation")
        if os.path.exists(aggregation_dir):
            args.output_file = os.path.join(aggregation_dir, "aggregated_results.json")
        else:
            args.output_file = os.path.join(args.input_dir, "aggregated_results.json")

    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_file}")

    # 1. Load data and build matrix rows.
    matrices = load_and_extract_matrices(args.input_dir)

    if not matrices['total']:
        print("❌ No valid evaluation result data found")
        return 1

    # 2. Run the aggregation.
    print("\n🔍 Running matrix computation with NumPy...")
    analysis = aggregate_all_metrics(matrices)

    # 3. Print and save the results.
    print_summary(analysis)
    save_aggregated_results(analysis, args.output_file)

    print("\n✅ Done!")
    return 0


if __name__ == "__main__":
    exit(main())
