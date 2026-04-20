"""
cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path array-gray/_generator/start_code.py \
    --function-path array-gray/_generator/log_solution.py \
    --input-path array-gray/_generator/test_cases/log_killer.txt \
    --time-limit 2.0
"""


def solve(n):
    if n == 0:
        print(0)
        return

    res = 0
    curr_n = n

    # While curr_n > 0, keep iterating according to the reflection rule.
    # Each iteration processes one highest-bit block.
    while curr_n > 0:
        # Find the highest-bit weight L (that is, 2^k) for the current n.
        # bit_length() returns the number of bits, so k = length - 1.
        k = curr_n.bit_length() - 1
        L = 1 << k

        # From the problem definition: A[L + i] = P[L - 1 - i] + L.
        # Let n = L + i, then i = n - L.
        # Therefore A[n] = A[L - 1 - (n - L)] + L.
        # Equivalently, A[n] = A[2*L - 1 - n] + L.

        # 1. Add the offset L contributed by the current block.
        res += L

        # 2. Map curr_n back to the reflection point in the previous block.
        # The next iteration processes A[2*L - 1 - curr_n].
        curr_n = (2 * L - 1) - curr_n

    print(res)
