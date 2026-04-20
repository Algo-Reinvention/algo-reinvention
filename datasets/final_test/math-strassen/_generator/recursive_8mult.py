"""
cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path math-strassen/_generator/start_code.py \
    --function-path math-strassen/_generator/recursive_8mult.py \
    --input-path math-strassen/_generator/test_cases/cubic_killer.txt \
    --time-limit 2.0
"""

"""
Correct but O(n^3): divide-and-conquer with 8 recursive multiplications.
"""


def _add(x, y):
    n = len(x)
    return [[x[i][j] + y[i][j] for j in range(n)] for i in range(n)]


def _mul(a, b, threshold=32):
    n = len(a)
    if n <= threshold:
        c = [[0] * n for _ in range(n)]
        for i in range(n):
            for k in range(n):
                aik = a[i][k]
                for j in range(n):
                    c[i][j] += aik * b[k][j]
        return c

    m = n // 2

    a11 = [row[:m] for row in a[:m]]
    a12 = [row[m:] for row in a[:m]]
    a21 = [row[:m] for row in a[m:]]
    a22 = [row[m:] for row in a[m:]]

    b11 = [row[:m] for row in b[:m]]
    b12 = [row[m:] for row in b[:m]]
    b21 = [row[:m] for row in b[m:]]
    b22 = [row[m:] for row in b[m:]]

    c11 = _add(_mul(a11, b11, threshold), _mul(a12, b21, threshold))
    c12 = _add(_mul(a11, b12, threshold), _mul(a12, b22, threshold))
    c21 = _add(_mul(a21, b11, threshold), _mul(a22, b21, threshold))
    c22 = _add(_mul(a21, b12, threshold), _mul(a22, b22, threshold))

    c = [[0] * n for _ in range(n)]
    for i in range(m):
        c[i][:m] = c11[i]
        c[i][m:] = c12[i]
        c[i + m][:m] = c21[i]
        c[i + m][m:] = c22[i]
    return c


def solve(n, a, b):
    c = _mul(a, b)
    out = sys.stdout.write
    for row in c:
        out(" ".join(map(str, row)) + "\n")


import sys
