"""

cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path array-moore/_generator/start_code.py \
    --function-path  array-moore/_generator/naive.py \
    --input-path array-moore/_generator/test_cases/n2e7.txt

"""

def solve(n, a):
    """
    Find the majority element using Boyer-Moore Voting Algorithm.
    :param n: Number of elements
    :param a: List of integers
    """
    # ================= Phase 1: Pick a candidate =================
    candidate = a[0]
    print(candidate)
