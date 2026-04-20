"""

cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path array-moore/_generator/start_code.py \
    --function-path  array-moore/_generator/moore_solution.py \
    --input-path array-moore/_generator/test_cases/n2e7.txt

"""

def solve(n, a):
    """
    Find the majority element using Boyer-Moore Voting Algorithm.
    :param n: Number of elements
    :param a: List of integers
    """
    # ================= Phase 1: Find the candidate =================
    candidate = None
    count = 0

    for x in a:
        if count == 0:
            # If the counter is zero, all previous votes have canceled out.
            # Set the current element as the new candidate.
            candidate = x
            count = 1
        elif x == candidate:
            # If the current element matches the candidate, increment the counter.
            count += 1
        else:
            # Otherwise a cancellation happens, so decrement the counter.
            count -= 1

    # ================= Phase 2: Verification =================
    # The problem guarantees that a majority element (count > n/2) exists,
    # so candidate must be the final answer here.
    #
    # Note: if the problem did not guarantee a majority element, you would need:
    # actual_count = 0
    # for x in a:
    #     if x == candidate:
    #         actual_count += 1
    # if actual_count > n // 2:
    #     print(candidate)
    # else:
    #     print("No majority element")

    # Output the result directly, as required by the problem.
    print(candidate)
