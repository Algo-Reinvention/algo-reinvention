"""
cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path math-strassen/_generator/start_code.py \
    --function-path math-strassen/_generator/cubic_naive.py \
    --input-path math-strassen/_generator/test_cases/cubic_killer.txt \
    --time-limit 2.0
"""

"""
Correct but O(n^3) classical matrix multiplication.
"""


def solve(n, a, b):
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            aik = a[i][k]
            for j in range(n):
                c[i][j] += aik * b[k][j]

    out = sys.stdout.write
    for row in c:
        out(" ".join(map(str, row)) + "\n")


import sys
