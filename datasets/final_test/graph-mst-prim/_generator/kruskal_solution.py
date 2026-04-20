def solve(n, m, adj):
    """
    n: number of nodes
    m: number of edges
    adj: adjacency list of the form adj[u] = [(v1, w1), (v2, w2), ...]
    """
    # 1. Extract all edges (u, v, w).
    # Because the graph is undirected, each edge appears twice in adj; keep only u < v.
    edges = []
    for u in range(1, n + 1):
        for v, w in adj[u]:
            if u < v:
                edges.append((u, v, w))
    
    # If n > 1 but there are no edges, the graph cannot be connected.
    if n > 1 and not edges:
        print("impossible")
        return

    # 2. Sort edges by ascending weight.
    edges.sort(key=lambda x: x[2])

    # 3. Initialize Union-Find.
    parent = list(range(n + 1))
    
    def find(i):
        # Path compression.
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    # 4. Scan edges in sorted order.
    mst_weight = 0
    edges_count = 0
    
    for u, v, w in edges:
        if union(u, v):
            mst_weight += w
            edges_count += 1
            # Once n - 1 edges are collected, the MST is complete.
            if edges_count == n - 1:
                break

    # 5. Decide the result.
    # An MST must contain n - 1 edges (except the n = 1 case, where it has 0).
    if n == 1:
        print(0)
    elif edges_count == n - 1:
        print(mst_weight)
    else:
        print("impossible")
