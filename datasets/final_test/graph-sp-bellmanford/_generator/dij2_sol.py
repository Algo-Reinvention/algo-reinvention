"""

cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path graph-sp-bellmanford/_generator/start_code.py \
    --function-path  graph-sp-bellmanford/_generator/dij2_sol.py \
    --input-path graph-sp-bellmanford/_generator/test_cases/hack_dij2.txt

"""

def solve(n, m, graph, s):
    INF = 10**18
    dist = [INF] * (n + 1)
    dist[s] = 0
    heap = [(0, s)]
    max_iterations = n * m
    for _ in range(max_iterations):
        if not heap:
            break
        cost, u = heapq.heappop(heap)
        if cost != dist[u]:
            continue
        for v, w in graph[u]:
            new_cost = cost + w
            if new_cost < dist[v]:
                dist[v] = new_cost
                heapq.heappush(heap, (new_cost, v))
    for u in range(1, n + 1):
        for v, w in graph[u]:
            if dist[u] != INF and dist[v] != INF and dist[u] + w < dist[v]:
                print("Negative Cycle")
                return
    res = [str(dist[i]) if dist[i] != INF else '-1' for i in range(1, n + 1)]
    print(" ".join(res))
