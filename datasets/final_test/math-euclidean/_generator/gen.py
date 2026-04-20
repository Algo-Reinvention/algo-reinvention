import random
import os
import sys
from io import StringIO
import contextlib

# Add the current directory to sys.path so euclidean.py can be imported.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from euclidean import solve

@contextlib.contextmanager
def capture_stdout():
    new_out = StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield new_out
    finally:
        sys.stdout = old_out

def generate_case(filename, a, b):
    # Match the problem format: one line with two integers.
    input_str = f"{a} {b}\n"
    with open(filename, 'w') as f:
        f.write(input_str)

    output_filename = filename.replace('.txt', '_groundtruth.txt')

    with open(output_filename, 'w') as f:
        with capture_stdout() as out:
            solve(a, b)
        f.write(out.getvalue().strip() + "\n")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases_dir = os.path.join(base_dir, 'test_cases')
    os.makedirs(test_cases_dir, exist_ok=True)

    # Case 1: random small case for correctness checking.
    generate_case(os.path.join(test_cases_dir, 'small_random.txt'),
                  random.randint(1, 1000), random.randint(1, 1000))

    # Case 2: subtraction-method killer.
    # One huge number and 1. Repeated subtraction needs about 10^18 steps and will TLE.
    # The Euclidean algorithm finishes in one modulo step: 10^18 % 1 = 0.
    a_big = 10**18 - 7
    b_small = 1
    generate_case(os.path.join(test_cases_dir, 'subtraction_killer.txt'), a_big, b_small)

    # Case 3: brute-force killer.
    # Two huge prime numbers. Brute force would scan downward from min(a, b)
    # for roughly 10^18 iterations and will TLE, while Euclid finishes in logarithmic time.
    p1 = 999999999999999989 # Large prime.
    p2 = 999999999999999997 # Large prime.
    generate_case(os.path.join(test_cases_dir, 'brute_force_killer.txt'), p1, p2)

    # Case 4: Fibonacci stress test, the classical worst case for Euclid.
    # Consecutive Fibonacci numbers maximize the number of Euclidean iterations.
    # Even so, values up to about 10^18 still need only around 90 steps.
    fib = [0, 1]
    for i in range(2, 92):
        fib.append(fib[i-1] + fib[i-2])
    generate_case(os.path.join(test_cases_dir, 'fibonacci_worst.txt'), fib[90], fib[91])

    # Case 5: Large Multiple
    # Two large numbers that share a large common divisor.
    common = 10**9 + 7
    generate_case(os.path.join(test_cases_dir, 'large_multiple.txt'),
                  common * 123456789, common * 987654321)

if __name__ == '__main__':
    main()
