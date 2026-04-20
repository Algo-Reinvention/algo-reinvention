"""

cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path graph-sp-floyd/_generator/start_code.py \
    --function-path  graph-sp-floyd/_generator/floyd1.py \
    --input-path graph-sp-floyd/_generator/test_cases/n300_dense.txt

"""

def solve(n, dist):
    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    for i in range(1, n + 1):
        line = []
        for j in range(1, n + 1):
            if dist[i][j] == float('inf'):
                line.append('INF')
            else:
                line.append(str(dist[i][j]))
        print(' '.join(line))
