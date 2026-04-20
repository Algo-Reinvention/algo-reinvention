def solve(n, dist):
    # Update the distance matrix using Triple-Loop relaxation
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i != j and i != k and j != k:
                    if dist[i][k] != float('inf') and dist[k][j] != float('inf') and dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
    return dist
