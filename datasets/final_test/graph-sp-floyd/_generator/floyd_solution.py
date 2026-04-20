def solve(n, dist):
    """
    n: number of nodes
    dist: initialized 2D distance matrix
    """
    INF = float('inf')
    
    # --- Core Floyd-Warshall algorithm ---
    # k is the intermediate node currently being considered.
    for k in range(1, n + 1):
        # i is the source node.
        for i in range(1, n + 1):
            # j is the destination node.
            for j in range(1, n + 1):
                # Only update if both partial paths exist.
                if dist[i][k] != INF and dist[k][j] != INF:
                    # Update when routing through k is shorter than the current path.
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
    
    # --- Output logic ---
    for i in range(1, n + 1):
        output_row = []
        for j in range(1, n + 1):
            if dist[i][j] == INF:
                output_row.append("INF")
            else:
                # Convert to string before joining.
                output_row.append(str(dist[i][j]))
        # Print one row at a time with space-separated entries.
        print(" ".join(output_row))
