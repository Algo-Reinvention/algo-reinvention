"""

cate=math-euclidean
cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path ${cate}/_generator/start_code.py \
    --function-path ${cate}/_generator/naive.py \
    --input-path ${cate}/_generator/test_cases/brute_force_killer.txt

"""

def solve(a, b):
    import math
    # Handle zero cases.
    if a == 0 or b == 0:
        print(max(a, b))
        return

    # Try divisors from the smaller number downward.
    limit = min(a, b)
    # Warning: this loop is extremely slow on large inputs.
    for i in range(limit, 0, -1):
        if a % i == 0 and b % i == 0:
            print(i)
            break
