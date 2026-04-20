"""
cd ${PROJECT_ROOT}/datasets/final_test/math-strassen/_generator
python gen.py
"""

import os
import sys
from io import StringIO
import contextlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strassen_solution import solve


def make_small_matrix(n, seed):
    x = seed & 0x7FFFFFFF
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        row = out[i]
        for j in range(n):
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            row[j] = (x % 201) - 100
    return out


def make_large_matrix(n, seed, bits=62):
    if bits < 2 or bits > 62:
        raise ValueError("bits must be in [2, 62]")

    x = seed & 0x7FFFFFFF
    out = [[0] * n for _ in range(n)]
    if bits == 62:
        for i in range(n):
            row = out[i]
            for j in range(n):
                x = (1103515245 * x + 12345) & 0x7FFFFFFF
                y = (1664525 * x + 1013904223) & 0x7FFFFFFF
                v = ((x << 31) ^ y) & ((1 << 62) - 1)
                row[j] = v - (1 << 61)
        return out

    mask = (1 << bits) - 1
    offset = 1 << (bits - 1)
    for i in range(n):
        row = out[i]
        for j in range(n):
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            y = (1664525 * x + 1013904223) & 0x7FFFFFFF
            v = ((x << 31) ^ y) & ((1 << 62) - 1)
            row[j] = (v & mask) - offset
    return out


@contextlib.contextmanager
def capture_stdout():
    new_out = StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield new_out
    finally:
        sys.stdout = old_out


def write_case(path, n, a, b):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write(" ".join(map(str, a[i])) + "\n")
        for i in range(n):
            f.write(" ".join(map(str, b[i])) + "\n")

    out_path = path.replace('.txt', '_groundtruth.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        with capture_stdout() as cap:
            solve(n, [row[:] for row in a], [row[:] for row in b])
        f.write(cap.getvalue())


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    tc_dir = os.path.join(base, 'test_cases')
    os.makedirs(tc_dir, exist_ok=True)

    n = 1
    a = [[7]]
    b = [[-3]]
    write_case(os.path.join(tc_dir, 'edge_n1.txt'), n, a, b)

    n = 2
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    write_case(os.path.join(tc_dir, 'small_n2.txt'), n, a, b)

    n = 64
    a = make_small_matrix(n, 114514)
    b = make_small_matrix(n, 1919810)
    write_case(os.path.join(tc_dir, 'random_n64.txt'), n, a, b)

    # Calibrated against execute_test:
    # - strassen_solution target_func_time < 1.8s
    # - O(n^3) baselines exceed 2.0s on this case
    n = 256
    a = make_large_matrix(n, 20260207, bits=60)
    b = make_large_matrix(n, 31415926, bits=60)
    write_case(os.path.join(tc_dir, 'cubic_killer.txt'), n, a, b)


if __name__ == '__main__':
    main()
