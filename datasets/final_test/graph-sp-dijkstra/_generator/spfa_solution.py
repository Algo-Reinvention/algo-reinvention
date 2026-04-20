def solve(n, m, graph, s):
    INF = float('inf')
    dist = [INF] * (n + 1)
    dist[s] = 0
    queue = collections.deque()
    queue.append(s)
    in_queue = [False] * (n + 1)
    in_queue[s] = True
    
    cnt = 1
    while queue:
        cnt += 1
        u = queue.popleft()
        in_queue[u] = False
        for (v, w) in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
    
    for i in range(1, n + 1):
        if dist[i] == INF:
            print(-1, end=' ')
        else:
            print(dist[i], end=' ')
    # print(cnt)