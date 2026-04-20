"""

cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path graph-sp-bellmanford/_generator/start_code.py \
    --function-path  graph-sp-bellmanford/_generator/dij3_sol.py \
    --input-path graph-sp-bellmanford/_generator/test_cases/hack_dij3.txt

"""

def solve(n, m, graph, s):
    INF = 10**18
    dist = [INF] * (n + 1)
    dist[s] = 0

    heap = [(0, s)]
    import heapq
    heapq.heapify(heap)

    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            new_d = d + w
            if new_d < dist[v]:
                dist[v] = new_d
                heapq.heappush(heap, (new_d, v))
                if v == s and new_d < 0:
                    print("Negative Cycle")
                    return

    for u in range(1, n + 1):
        for v, w in graph[u]:
            if dist[u] < INF and dist[u] + w < dist[v]:
                print("Negative Cycle")
                return

    print(" " .join(str(dist[i]) if dist[i] < INF else "-1" for i in range(1, n + 1)))
