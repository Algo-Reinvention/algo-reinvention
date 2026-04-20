"""
cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path array-gray/_generator/start_code.py \
    --function-path array-gray/_generator/incremental_gray.py \
    --input-path array-gray/_generator/test_cases/linear_killer.txt \
    --time-limit 2.0
"""

"""
Correct but O(n): incrementally walk BRGC values with lowbit flips.
"""


def solve(n):
    g = 0
    i = 1
    while i <= n:
        g ^= (i & -i)
        i += 1
    print(g)
