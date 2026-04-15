#!/usr/bin/env python3
"""
Aggregation script for evaluation results (Code Generation / Execution)
Usage:
python aggregate_unlearn.py <input_dir> [keyword]
"""

import re
import os
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np

# --- Data loading and preprocessing ---

def extract_problem_id(file_path) -> str:
    try:
        filename = Path(file_path).name
        return filename.split('.')[0]
    except Exception:
        return "unknown_id"

def natural_key(text: str):
    # Ensure problem_2 appears before problem_10.
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', text)]

def load_evaluation_results(output_dir: str) -> List[Dict[str, Any]]:
    results = []
    print(f"Loading evaluation results from {output_dir}...")

    if not os.path.exists(output_dir):
        print(f"❌ Error: directory does not exist: {output_dir}")
        return []

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if 'generated_responses' in data and \
                       isinstance(data['generated_responses'], list) and \
                       len(data['generated_responses']) > 0:

                        response = data['generated_responses'][0]
                        data['problem_id'] = extract_problem_id(file_path)
                        # Save the file path so later stages can search the file content.
                        data['file_full_path'] = file_path

                        is_correct = int(response.get('final_correctness', 0))
                        is_solved = is_correct == 1
                        final_status = response.get('final_status', 'unknown')
                        tmp_messages = response.get('final_message_history', [])
                        num_turns = sum(1 for msg in tmp_messages if msg.get('role') == 'assistant')

                        data['summary'] = {
                            'is_solved': is_solved,
                            'final_status': final_status,
                            'num_turns': num_turns
                        }
                        results.append(data)
                except Exception:
                    continue

    print(f"Successfully loaded {len(results)} evaluation results")
    return results


# --- Core performance analysis ---

def analyze_performance_by_status(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_problems = len(results)
    if total_problems == 0:
        return {}

    status_counts = defaultdict(int)
    successful_turns = []
    target_statuses = ['correct', 'max_tokens_reached', 'submitted_incorrect']

    for result in results:
        summary = result.get('summary', {})
        final_status = summary.get('final_status', 'unknown')
        num_turns = summary.get('num_turns', 0)
        status_counts[final_status] += 1
        if final_status == 'correct':
            successful_turns.append(num_turns)

    status_proportions = {status: status_counts.get(status, 0) / total_problems for status in target_statuses}
    avg_successful_turns = np.mean(successful_turns) if successful_turns else 0.0

    return {
        'total_problems': total_problems,
        'status_proportions': status_proportions,
        'successful_turns_data': {
            'count': len(successful_turns),
            'avg_turns': avg_successful_turns,
        }
    }


def generate_detailed_report(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    detailed_report = []
    for result in results:
        summary = result.get('summary', {})
        detailed_report.append({
            'problem_id': result.get('problem_id', 'unknown'),
            'file_full_path': result.get('file_full_path', ''), # Pass the path through the report.
            'problem_solved': summary.get('is_solved', False),
            'final_status': summary.get('final_status', 'unknown'),
            'num_turns': summary.get('num_turns', 0)
        })
    detailed_report.sort(key=lambda x: (x['problem_solved'], x['num_turns']), reverse=True)
    return detailed_report


# --- Result printing and keyword search ---

def print_correct_and_check_keyword(detailed_report: List[Dict[str, Any]], keyword: str = None):
    """
    1. Print all correct problem IDs.
    2. If a keyword is provided, check whether the content of each correct file contains it.
    """
    correct_items = sorted(
        [r for r in detailed_report if r['problem_solved']],
        key=lambda r: natural_key(r['problem_id'])
    )
    correct_ids = [r['problem_id'] for r in correct_items]

    print("\n" + "-"*40)
    print(f"🎯 Correct file list (Total: {len(correct_ids)})")
    print("-"*40)

    if not correct_ids:
        print("  (No correct results)")
    else:
        for i in range(0, len(correct_ids), 2):
            line = correct_ids[i:i+2]
            print("  " + "    ".join(f"• {name:<30}" for name in line))

    # --- Keyword search logic ---
    if keyword:
        print("\n" + "🔍" + " File Content Keyword Search " + "🔍")
        print(f"Target keyword: '{keyword}'")

        matches = []
        for item in correct_items:
            file_path = item.get('file_full_path')
            if not file_path or not os.path.exists(file_path):
                continue

            try:
                # Read the file content and search it directly.
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if keyword in content:
                        matches.append(item['problem_id'])
            except Exception as e:
                print(f"  ⚠️ Unable to read file {file_path}: {e}")

        if matches:
            print(f"✅ Match found! The keyword appears in the **file content** of these correct results:")
            for m in matches:
                print(f"  -> {m}")
        else:
            print(f"❌ The keyword '{keyword}' was not found in the file content of any correct result.")
        print("-" * 40)


def print_analysis_summary(analysis: Dict[str, Any]):
    if not analysis: return
    print("\n" + "="*60)
    print("Algorithm Evaluation Aggregation Report")
    print("="*60)

    total_problems = analysis['total_problems']
    status_props = analysis['status_proportions']
    success_turns = analysis['successful_turns_data']

    print(f"\n📊 Total problems: {total_problems}")
    print("\n📌 Final status distribution:")
    print(f"  ✅ Correct (passed): {status_props.get('correct', 0.0):.1%}")
    print(f"  🛑 Max Tokens Reached: {status_props.get('max_tokens_reached', 0.0):.1%}")
    print(f"  🔁 Max Turn Reached: {status_props.get('submitted_incorrect', 0.0):.1%}")
    print("-" * 40)
    print(f"⭐ Average turn count for successful responses ({success_turns['count']} items): {success_turns['avg_turns']:.2f}")


def save_aggregated_results(analysis: Dict[str, Any], detailed_report: List[Dict[str, Any]], output_file: str):
    aggregated_data = {
        'analysis': analysis,
        'detailed_report': detailed_report,
        'metadata': {
            'total_problems_analyzed': len(detailed_report),
            'analysis_timestamp': __import__('time').time()
        }
    }
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Aggregated results saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_unlearn.py <input_dir> [keyword]")
        return 1

    input_dir = sys.argv[1]
    keyword = sys.argv[2] if len(sys.argv) > 2 else None

    output_file = os.path.join(input_dir, "aggregation/aggregated_results.json")

    results = load_evaluation_results(input_dir)
    if not results:
        print("❌ No valid evaluation result files found")
        return 1

    analysis = analyze_performance_by_status(results)
    detailed_report = generate_detailed_report(results)

    # 1. Print the basic summary.
    print_analysis_summary(analysis)

    # 2. Print the correct IDs and optionally run keyword search on file content.
    print_correct_and_check_keyword(detailed_report, keyword)

    # 3. Save the aggregated report.
    save_aggregated_results(analysis, detailed_report, output_file)
    print("\n✅ Aggregation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
