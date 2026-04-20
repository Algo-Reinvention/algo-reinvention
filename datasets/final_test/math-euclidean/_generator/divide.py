"""

cate=math-euclidean
cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path ${cate}/_generator/start_code.py \
    --function-path ${cate}/_generator/divide.py \
    --input-path ${cate}/_generator/test_cases/subtraction_killer.txt

"""

def solve(a, b):
    # Handle cases involving zero.
    if a == 0:
        print(b)
        return
    if b == 0:
        print(a)
        return

    # Keep subtracting until the two numbers become equal.
    # Warning: this becomes extremely slow when a and b are large and far apart.
    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a
    print(a)
