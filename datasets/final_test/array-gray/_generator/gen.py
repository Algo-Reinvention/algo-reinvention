"""
cd ${PROJECT_ROOT}/datasets/final_test/array-gray/_generator
python gen.py
"""

import os
import sys
from io import StringIO
import contextlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gray_solution import solve

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


@contextlib.contextmanager
def capture_stdout():
    new_out = StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield new_out
    finally:
        sys.stdout = old_out


def generate_case(filename, n):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")

    output_filename = filename.replace(".txt", "_groundtruth.txt")
    with open(output_filename, "w", encoding="utf-8") as f:
        with capture_stdout() as out:
            solve(n)
        f.write(out.getvalue().strip() + "\n")


def alternating_bits_number(bits):
    # Build 101010... (length=bits) to maximize the iteration count
    # of reflection-based O(log n) decoders.
    n = 0
    for i in range(bits):
        n = (n << 1) | (1 if i % 2 == 0 else 0)
    return n


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    case_dir = os.path.join(base_dir, "test_cases")
    os.makedirs(case_dir, exist_ok=True)

    generate_case(os.path.join(case_dir, "edge_zero.txt"), 0)
    generate_case(os.path.join(case_dir, "small_index.txt"), 7)
    generate_case(os.path.join(case_dir, "large_64bit.txt"), 987654321012345678)

    # Designed to make O(n) solutions timeout around the 2s limit under execute_test.
    generate_case(os.path.join(case_dir, "linear_killer.txt"), 350000000)

    # Designed to make reflection-based O(log n) decoders timeout around 2s
    # while the direct O(1) formula remains fast.
    generate_case(os.path.join(case_dir, "log_killer.txt"), alternating_bits_number(260000))


if __name__ == "__main__":
    main()
