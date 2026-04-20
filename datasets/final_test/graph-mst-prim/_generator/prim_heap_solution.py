def solve(n, m, adj):
    """
    n: number of nodes
    m: number of edges
    adj: adjacency list, where adj[u] stores (v, weight)
    """
    # Track whether each node has already been added to the spanning tree.
    visited = [False] * (n + 1)
    # Min-heap storing (weight, target_node).
    # Start from node 1, though any node would work.
    pq = [(0, 1)] 
    
    total_weight = 0
    count = 0  # Number of nodes already added to the spanning tree.
    
    while pq:
        # Pop the edge currently closest to the spanning tree.
        w, u = heapq.heappop(pq)
        
        # Skip if the node is already inside the spanning tree.
        if visited[u]:
            continue
        
        # Add the node to the spanning tree.
        visited[u] = True
        total_weight += w
        count += 1
        
        # Visit all neighbors of the current node.
        for v, weight in adj[u]:
            if not visited[v]:
                # Push unvisited neighbors into the heap; it stays ordered by weight.
                heapq.heappush(pq, (weight, v))
                
    # If count == n, the graph is connected.
    if count == n:
        print(total_weight)
    else:
        # Otherwise the graph is disconnected, so no MST exists.
        print("impossible")
