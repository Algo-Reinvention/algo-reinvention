"""

cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path graph-sp-bellmanford/_generator/start_code.py \
    --function-path  graph-sp-bellmanford/_generator/bf_sol.py \
    --input-path graph-sp-bellmanford/_generator/test_cases/hack_dij4.txt > ${PROJECT_ROOT}/datasets/final_test/graph-sp-bellmanford/_generator/test_cases/hack_dij4_groundtruth.txt

"""


def solve(n, m, graph, s):
    # 1. Initialize the distance array.
    # Use float('inf') to represent infinity.
    dist = [float('inf')] * (n + 1)
    dist[s] = 0

    # 2. Perform n - 1 rounds of relaxation.
    for i in range(n - 1):
        updated = False
        for u in range(1, n + 1):
            # Skip outgoing edges from unreachable nodes.
            if dist[u] == float('inf'):
                continue

            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    updated = True

        # Optimization: if nothing changed in this round, the optimum is already found.
        if not updated:
            break

    # 3. Use the nth round to detect a negative cycle.
    has_negative_cycle = False
    for u in range(1, n + 1):
        if dist[u] == float('inf'):
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                # A further relaxation is still possible, so a negative cycle exists.
                has_negative_cycle = True
                break
        if has_negative_cycle:
            break

    # 4. Print the result in the required format.
    if has_negative_cycle:
        print("Negative Cycle")
    else:
        results = []
        for i in range(1, n + 1):
            if dist[i] == float('inf'):
                results.append("-1")
            else:
                results.append(str(dist[i]))
        print(" ".join(results))
