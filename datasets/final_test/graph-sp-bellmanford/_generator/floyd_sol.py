"""

cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path graph-sp-bellmanford/_generator/start_code.py \
    --function-path  graph-sp-bellmanford/_generator/floyd_sol.py \
    --input-path graph-sp-bellmanford/_generator/test_cases/n500.txt

"""


def solve(n, m, graph, s):
    # 1. Initialize the 2D distance matrix.
    # dist[i][j] stores the distance from i to j.
    inf = float('inf')
    dist = [[inf] * (n + 1) for _ in range(n + 1)]

    # The distance from a node to itself is 0.
    for i in range(1, n + 1):
        dist[i][i] = 0

    # 2. Fill graph edges into the matrix.
    # If multiple edges exist, keep the shortest one.
    for u in range(1, n + 1):
        for v, w in graph[u]:
            if w < dist[u][v]:
                dist[u][v] = w

    # 3. Core Floyd-Warshall triple loop.
    # k is the intermediate node, i is the source, and j is the destination.
    # Note: k must be the outermost loop.
    for k in range(1, n + 1):
        # Optimization: skip this branch if i cannot reach k.
        for i in range(1, n + 1):
            if dist[i][k] == inf:
                continue
            for j in range(1, n + 1):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # 4. Detect negative cycles.
    # The problem guarantees none, but this is still the standard check.
    has_negative_cycle = False
    for i in range(1, n + 1):
        if dist[i][i] < 0:
            has_negative_cycle = True
            break

    # 5. Print the result.
    if has_negative_cycle:
        print("Negative Cycle")
    else:
        # Only the row for source s is needed.
        source_row = dist[s]
        results = []
        for i in range(1, n + 1):
            if source_row[i] == inf:
                results.append("-1")
            else:
                results.append(str(source_row[i]))
        print(" ".join(results))
