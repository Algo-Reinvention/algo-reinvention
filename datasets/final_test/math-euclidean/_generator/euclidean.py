"""

cate=math-euclidean
cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path ${cate}/_generator/start_code.py \
    --function-path ${cate}/_generator/euclidean.py \
    --input-path ${cate}/_generator/test_cases/num.txt

"""

def solve(a, b):
    # Implement the Euclidean algorithm iteratively.
    while b != 0:
        a, b = b, a % b
    print(a)
