import random
import os
import sys
import string
# Add current directory to path so we can import the reference solution.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kmp_sol import solve

from io import StringIO
import contextlib

@contextlib.contextmanager
def capture_stdout():
    new_out = StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield new_out
    finally:
        sys.stdout = old_out

def generate_case(filename, text, pattern):
    input_str = f"{text}\n{pattern}\n"
    with open(filename, 'w') as f:
        f.write(input_str)
    
    output_filename = filename.replace('.txt', '_groundtruth.txt')
    
    with open(output_filename, 'w') as f:
        with capture_stdout() as out:
            solve(text, pattern)
        f.write(out.getvalue())

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases_dir = os.path.join(base_dir, 'test_cases')
    os.makedirs(test_cases_dir, exist_ok=True)

    # Case 1: Random
    t = ''.join(random.choices(string.ascii_uppercase, k=10000))
    p = ''.join(random.choices(string.ascii_uppercase, k=100))
    generate_case(os.path.join(test_cases_dir, 'random.txt'), t, p)

    # Case 2: Naive Killer
    # O(N*M) check.
    # Text length 500,000, Pattern length 50,000.
    # Naive Operations ~= 5*10^5 * 5*10^4 = 2.5 * 10^10. 
    # This will definitely timeout (requires >10s in C++, much more in Python).
    # KMP O(N) ~= 5.5 * 10^5 operations. Instant.
    t = 'A' * 500000 + 'B'
    p = 'A' * 50000 + 'B'
    generate_case(os.path.join(test_cases_dir, 'naive_killer.txt'), t, p)

    # Case 3: Many matches
    # Testing lots of output I/O and state transitions.
    t = 'A' * 200000
    p = 'A' * 100
    generate_case(os.path.join(test_cases_dir, 'many_matches.txt'), t, p)

if __name__ == '__main__':
    main()
