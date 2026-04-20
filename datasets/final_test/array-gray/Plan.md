# array-gray Completion Plan

## Task Understanding

- Target directory: `algo_test/datasets/final_test/array-gray`
- Goal: provide a complete final-test dataset category for computing the **0-based n-th BRGC value**.
- Evaluation intent: emphasize whether models can derive an O(1) solution rather than scanning/generating sequence prefixes.

## Deliverables

1. `_generator/` scripts:
- `start_code.py`
- `gray_solution.py` (O(1) reference)
- `linear_scan.py` and `incremental_gray.py` (correct but non-O(1) baselines)
- `gen.py` and generated `test_cases/*` with groundtruth

2. Prompt set:
- `level0/0..7.json`
- `level1/0..7.json`
- `level2/0..7.json`

3. Consistency rules:
- `1..4` variants explicitly include design/invent requirement
- `0,5,6,7` do not explicitly include design/invent wording

## Verification Checklist

- test cases generated successfully
- all level JSON files parse successfully
- `gray_solution.py` passes killer case within limit
- non-O(1) baselines timeout on killer case under same limit
