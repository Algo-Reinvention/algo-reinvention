"""
cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path array-gray/_generator/start_code.py \
    --function-path array-gray/_generator/gray_solution.py \
    --input-path array-gray/_generator/test_cases/linear_killer.txt \
    --time-limit 2.0
"""


def solve(n):
    print(n ^ (n >> 1))
