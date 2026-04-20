def solve(n, m, adj):
    """
    O(n^2) Prim's Algorithm
    Suitable for dense graphs.
    """
    inf = float('inf')
    # min_dist[i] stores the lightest edge connecting node i to the current tree.
    min_dist = [inf] * (n + 1)
    # visited[i] marks whether node i is already in the spanning tree.
    visited = [False] * (n + 1)
    
    # Start from node 1, though any starting node would work.
    min_dist[1] = 0
    total_weight = 0
    nodes_added = 0

    for _ in range(n):
        # 1. Find the unvisited node u closest to the current tree.
        u = -1
        for i in range(1, n + 1):
            if not visited[i]:
                if u == -1 or min_dist[i] < min_dist[u]:
                    u = i
        
        # 2. If no reachable node exists, the graph is disconnected.
        if u == -1 or min_dist[u] == inf:
            print("impossible")
            return
        
        # 3. Add u to the tree and accumulate its edge weight.
        visited[u] = True
        total_weight += min_dist[u]
        nodes_added += 1
        
        # 4. Update each neighbor's best connection to the tree using u.
        # For extremely dense graphs, an adjacency matrix scan can be faster,
        # but this template provides an adjacency list, so we iterate over adj[u].
        for v, weight in adj[u]:
            if not visited[v] and weight < min_dist[v]:
                min_dist[v] = weight

    # If all n nodes were added, output the total weight.
    if nodes_added == n:
        print(total_weight)
    else:
        print("impossible")
