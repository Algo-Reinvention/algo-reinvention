"""
cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path math-strassen/_generator/start_code.py \
    --function-path math-strassen/_generator/strassen_solution.py \
    --input-path math-strassen/_generator/test_cases/cubic_killer.txt \
    --time-limit 2.0
"""


def _add(a, b):
    n = len(a)
    return [[a[i][j] + b[i][j] for j in range(n)] for i in range(n)]


def _sub(a, b):
    n = len(a)
    return [[a[i][j] - b[i][j] for j in range(n)] for i in range(n)]


def _naive(a, b):
    n = len(a)
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        ci = c[i]
        ai = a[i]
        for k in range(n):
            aik = ai[k]
            bk = b[k]
            for j in range(n):
                ci[j] += aik * bk[j]
    return c


def _strassen(a, b, threshold=32):
    n = len(a)
    if n <= threshold:
        return _naive(a, b)

    m = n // 2

    a11 = [row[:m] for row in a[:m]]
    a12 = [row[m:] for row in a[:m]]
    a21 = [row[:m] for row in a[m:]]
    a22 = [row[m:] for row in a[m:]]

    b11 = [row[:m] for row in b[:m]]
    b12 = [row[m:] for row in b[:m]]
    b21 = [row[:m] for row in b[m:]]
    b22 = [row[m:] for row in b[m:]]

    m1 = _strassen(_add(a11, a22), _add(b11, b22), threshold)
    m2 = _strassen(_add(a21, a22), b11, threshold)
    m3 = _strassen(a11, _sub(b12, b22), threshold)
    m4 = _strassen(a22, _sub(b21, b11), threshold)
    m5 = _strassen(_add(a11, a12), b22, threshold)
    m6 = _strassen(_sub(a21, a11), _add(b11, b12), threshold)
    m7 = _strassen(_sub(a12, a22), _add(b21, b22), threshold)

    c11 = _add(_sub(_add(m1, m4), m5), m7)
    c12 = _add(m3, m5)
    c21 = _add(m2, m4)
    c22 = _add(_sub(_add(m1, m3), m2), m6)

    c = [[0] * n for _ in range(n)]
    for i in range(m):
        c[i][:m] = c11[i]
        c[i][m:] = c12[i]
        c[i + m][:m] = c21[i]
        c[i + m][m:] = c22[i]
    return c


def solve(n, a, b):
    c = _strassen(a, b)
    out = sys.stdout.write
    for row in c:
        out(" ".join(map(str, row)) + "\n")


import sys
