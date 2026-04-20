"""

cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path graph-sp-bellmanford/_generator/start_code.py \
    --function-path  graph-sp-bellmanford/_generator/dij4_sol.py \
    --input-path graph-sp-bellmanford/_generator/test_cases/hack_dij4.txt

"""

def solve(n, m, graph, s):
    INF = 10**18
    dist = [INF] * (n + 1)
    dist[s] = 0
    heap = [(0, s)]
    processed = [0] * (n + 1)

    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        processed[u] += 1
        if processed[u] > n:
            print("Negative Cycle")
            return
        for v, w in graph[u]:
            new_d = d + w
            if new_d < dist[v]:
                dist[v] = new_d
                heapq.heappush(heap, (new_d, v))

    result = []
    for i in range(1, n + 1):
        if dist[i] == INF:
            result.append(str(-1))
        else:
            result.append(str(dist[i]))
    print(" ".join(result))
