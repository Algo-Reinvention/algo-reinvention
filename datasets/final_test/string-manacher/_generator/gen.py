import random
import os
import sys
import string
# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sol import solve

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

def generate_case(filename, s):
    with open(filename, 'w') as f:
        f.write(s)
    
    output_filename = filename.replace('.txt', '_groundtruth.txt')
    
    with open(output_filename, 'w') as f:
        with capture_stdout() as out:
            solve(s)
        f.write(out.getvalue())

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases_dir = os.path.join(base_dir, 'test_cases')
    os.makedirs(test_cases_dir, exist_ok=True)

    # Case 1: Random
    # N = 500,000.
    # Checks efficiency on average cases. O(N^2) might pass small randoms, but fails 500k.
    s = ''.join(random.choices(string.ascii_lowercase, k=500000))
    generate_case(os.path.join(test_cases_dir, 'random.txt'), s)

    # Case 2: All same (Killer for naive expansion check) O(N^2)
    # 200,000 a's -> 2*10^10 operations for naive.
    # Manacher O(N) = 2*10^5 operations.
    s = 'a' * 200000
    generate_case(os.path.join(test_cases_dir, 'all_a.txt'), s)

    # Case 3: Alternating (Killer for some heuristics)
    s = 'ab' * 100000
    generate_case(os.path.join(test_cases_dir, 'alternating.txt'), s)

if __name__ == '__main__':
    main()
