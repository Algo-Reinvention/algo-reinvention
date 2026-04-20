"""

cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path graph-sp-bellmanford/_generator/start_code.py \
    --function-path  graph-sp-bellmanford/_generator/dij1_sol.py \
    --input-path graph-sp-bellmanford/_generator/test_cases/hack_dij1.txt

"""

def solve(n, m, graph, s):
    INF = 10**18
    dist = [INF] * (n + 1)
    dist[s] = 0
    heap = [(0, s)]
    max_iters = n * m

    for _ in range(max_iters):
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
            if dist[u] + w < dist[v]:
                print("Negative Cycle")
                return

    res = []
    for i in range(1, n + 1):
        if dist[i] == INF:
            res.append("-1")
        else:
            res.append(str(dist[i]))
    print(" ".join(res))
