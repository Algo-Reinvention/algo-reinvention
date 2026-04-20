def solve(n, m, graph, s):
    INF = 10**18
    dist = [INF] * (n + 1)
    dist[s] = 0
    visited = [False] * (n + 1)

    for _ in range(n):
        u = -1
        min_val = INF
        for i in range(1, n + 1):
            if not visited[i] and dist[i] < min_val:
                min_val = dist[i]
                u = i
        if u == -1:
            break
        visited[u] = True
        for v, w in graph[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist

    output_list = []
    for i in range(1, n + 1):
        if dist[i] == INF:
            output_list.append("-1")
        else:
            output_list.append(str(dist[i]))
    print(" ".join(output_list))
