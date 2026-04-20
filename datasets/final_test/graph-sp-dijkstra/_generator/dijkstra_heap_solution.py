def solve(n, m, graph, s):
    INF = float('inf')
    dist = [INF] * (n+1)
    dist[s] = 0
    heap = [(0, s)]
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    result = []
    for i in range(1, n+1):
        if dist[i] == INF:
            result.append(-1)
        else:
            result.append(dist[i])
    
    print(' '.join(map(str, result)))
