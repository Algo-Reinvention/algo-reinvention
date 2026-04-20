from collections import deque

def solve(n, dist):
    """
    n: number of nodes
    dist: initialized 2D distance matrix, where dist[i][j] = edge_weight or INF
    """
    INF = float('inf')
    
    # 1. SPFA runs best on adjacency lists, so convert the matrix first.
    # Only keep actual edges where dist[i][j] != INF and i != j.
    adj = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and dist[i][j] != INF:
                adj[i].append((j, dist[i][j]))

    # 2. Prepare the final result matrix to store all n SPFA runs.
    final_results = [[INF] * (n + 1) for _ in range(n + 1)]

    # 3. Run single-source SPFA once per start node.
    for start_node in range(1, n + 1):
        # --- Core single-source SPFA logic ---
        d = [INF] * (n + 1)
        in_queue = [False] * (n + 1)
        
        d[start_node] = 0
        queue = deque([start_node])
        in_queue[start_node] = True
        
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            
            for v, weight in adj[u]:
                if d[v] > d[u] + weight:
                    d[v] = d[u] + weight
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        
        # Store the current SPFA result into the final matrix.
        for target_node in range(1, n + 1):
            final_results[start_node][target_node] = d[target_node]

    # 4. Print the result.
    for i in range(1, n + 1):
        row_output = []
        for j in range(1, n + 1):
            val = final_results[i][j]
            if val == INF:
                row_output.append("INF")
            else:
                row_output.append(str(val))
        print(" ".join(row_output))
