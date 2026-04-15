#!/usr/bin/env python3
"""
cd ${PROJECT_ROOT}/
python simple_parallel/aggregation/aggregate_forget.py \
    --input_dir _output/results/qwen3-4b-unlearn/think/ablation-1214/baseline-len2048-lr2e5-rl0.3-dr3/global_step_1/benchmarks/math500-3 \
    --target_str ijkstra
    Scoring rule: if the response contains target_str, score 0 (failure);
    otherwise score 1 (success).
"""

import json
import argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate forgetting performance")
    parser.add_argument("--input_dir", type=str, default="./_output", help="Input directory")
    parser.add_argument("--output_file", type=str, help="Output file path")
    parser.add_argument("--show_details", action="store_true", help="Show detailed report")
    
    # target_str is required.
    parser.add_argument("--target_str", type=str, required=True, 
                       help="String to detect (for example, the forgetting target). Present = 0, absent = 1.")
    
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_file = Path(args.output_file) if args.output_file else input_path / "aggregation/aggregated_results.json"

    print(f"📂 Loading directory: {input_path}")
    print(f"🎯 Forgetting target: '{args.target_str}'")
    print(f"ℹ️  Scoring rule: target string present = 0, absent = 1")

    # --- 1. Scan data and compute scores ---
    results_matrix = []  # Stores the 0/1 matrix (rows=problems, cols=responses).
    problem_details = [] 
    max_k = 0            # Maximum number of responses.

    json_files = list(input_path.rglob("*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'generated_responses' not in data:
                continue

            # --- Core scoring logic ---
            # 1 = clean (forgetting succeeded), 0 = leaked (forgetting failed)
            row_scores = []
            for resp in data['generated_responses']:
                # Support both dictionary-style and string-style responses.
                content = resp['response'] if isinstance(resp, dict) else str(resp)
                
                if args.target_str in content:
                    row_scores.append(0) # Contains the target string: failure.
                else:
                    row_scores.append(1) # Clean: success.

            if not row_scores: continue

            max_k = max(max_k, len(row_scores))
            results_matrix.append(row_scores)
            
            # Collect per-problem statistics.
            mean_score = np.mean(row_scores)
            problem_details.append({
                'id': data.get('problem_id', file_path.stem),
                'score': mean_score,
                'clean_count': sum(row_scores),
                'total_count': len(row_scores)
            })

        except Exception as e:
            print(f"⚠️  Skipping file {file_path.name}: {e}")

    total_problems = len(results_matrix)
    if total_problems == 0:
        print("❌ No valid data found")
        return

    # --- 2. Compute summary statistics (NumPy) ---
    
    # Pad rows to max_k with 0. This keeps the existing behavior where
    # missing generations are treated as failures.
    padded_matrix = np.array([row + [0] * (max_k - len(row)) for row in results_matrix])

    # Compute the forgetting success rate (clean rate) at each position.
    success_rate_per_k = padded_matrix.mean(axis=0)
    
    final_mean = success_rate_per_k.mean()
    final_std = success_rate_per_k.std()

    # --- 3. Print the report ---

    print("\n" + "="*60)
    print("🧹 Forgetting Evaluation Results (Clean Rate)")
    print("="*60)
    print(f"  Total samples: {total_problems}")
    
    acc_dict = {f"{k+1}": float(v) for k, v in enumerate(success_rate_per_k)}
    
    for k, rate in acc_dict.items():
        print(f"  Pos@{k}: {rate:.1%} (probability that target_str is absent at this position)")

    print("-" * 40)
    print(f"⭐ Mean forgetting success rate: {final_mean:.1%}")
    print(f"  Variability (Std): ±{final_std:.1%}")

    # --- 4. Save the report ---
    output_data = {
        'config': {
            'target_str': args.target_str,
            'metric': 'forget_success_rate (1=clean, 0=leaked)'
        },
        'metrics': {
            'total_samples': total_problems,
            'position_success_rates': acc_dict,
            'mean': float(final_mean),
            'std': float(final_std)
        },
        'details': sorted_details if 'sorted_details' in locals() else problem_details
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
