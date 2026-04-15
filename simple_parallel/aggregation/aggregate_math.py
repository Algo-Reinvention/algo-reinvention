#!/usr/bin/env python3
"""
Math evaluation result aggregation script.

cd ${PROJECT_ROOT}/
python simple_parallel/aggregation/aggregate_math.py \
    --input_dir _output/results/qwen3-4b-unlearn/think/ablation-1214/baseline-len2048-lr2e5-rl0.3-dr3/global_step_1/benchmarks/math500-3

"""

import json
import argparse
import numpy as np
from pathlib import Path
from math_verify import parse, verify

def main():
    parser = argparse.ArgumentParser(description="Aggregate math evaluation results")
    parser.add_argument("--input_dir", type=str, default="./_output", help="Input directory")
    parser.add_argument("--output_file", type=str, help="Output file path")
    parser.add_argument("--show_details", action="store_true", help="Show detailed report")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    # Select a default output path automatically.
    output_file = Path(args.output_file) if args.output_file else input_path / "aggregation/aggregated_results.json"

    print(f"Loading and analyzing results from {input_path}...")

    # --- 1. Load data and build the result matrix ---
    results_matrix = []  # Stores lists of 0/1 outcomes.
    problem_details = [] # Stores per-problem details for reporting.
    max_k = 0            # Tracks the maximum number of responses.

    # Find all JSON files.
    json_files = list(input_path.rglob("*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'solution' not in data or 'generated_responses' not in data:
                continue

            # Parse the reference answer.
            solution_parsed = parse(f"\\boxed{{{data['solution']}}}")
            
            # Compute the correctness vector for the current problem: [1, 0, 1, ...]
            # Each response is checked independently and mapped to 1 (True) or 0 (False).
            row_correctness = [
                1 if verify(parse(resp['response']), solution_parsed) else 0
                for resp in data['generated_responses']
            ]

            if not row_correctness: continue

            # Update the maximum response count (K).
            max_k = max(max_k, len(row_correctness))
            
            results_matrix.append(row_correctness)
            
            # Collect detailed information for the report.
            problem_details.append({
                'id': data.get('problem_id', file_path.stem),
                'mean': np.mean(row_correctness),
                'count': sum(row_correctness)
            })

        except Exception as e:
            print(f"Error while processing {file_path.name}: {e}")

    total_problems = len(results_matrix)
    if total_problems == 0:
        print("❌ No valid data found")
        return

    # --- 2. Core computation (vectorized with NumPy) ---
    
    # Normalize the matrix by padding rows to max_k with 0.
    # This matches the original behavior where missing responses count as errors.
    padded_matrix = np.array([row + [0] * (max_k - len(row)) for row in results_matrix])

    # axis=0 computes the column-wise mean -> Acc@1, Acc@2, ... Acc@K
    acc_per_k = padded_matrix.mean(axis=0)
    
    # Compute the mean and standard deviation across Acc values.
    final_mean = acc_per_k.mean()
    final_std = acc_per_k.std()

    # --- 3. Print the report ---

    print("\n" + "="*60)
    print("📊 Math Evaluation Independent-Response Accuracy Report")
    print("="*60)
    print(f"  Total problems: {total_problems}")
    print(f"  Maximum response count (K): {max_k}")
    
    # Format the Acc@k dictionary.
    acc_dict = {f"{k+1}": float(v) for k, v in enumerate(acc_per_k)}
    
    for k, acc in acc_dict.items():
        print(f"  Acc@{k}: {acc:.1%}")

    print("-" * 40)
    print(f"⭐ Mean Acc@k: {final_mean:.1%}")
    print(f"  Acc@k standard deviation: ±{final_std:.1%}")

    # --- 4. Save results ---
    output_data = {
        'metrics': {
            'total_problems': total_problems,
            'response_accuracies': acc_dict,
            'mean': float(final_mean),
            'std': float(final_std)
        },
        'problem_details': sorted_details if 'sorted_details' in locals() else problem_details
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
