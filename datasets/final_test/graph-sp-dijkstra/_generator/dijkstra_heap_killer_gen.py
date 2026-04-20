from pathlib import Path


DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parent / "test_cases" / "dijkstra_heap_killer.txt"


def generate_graph(n, source, filename=None):
    output_path = Path(filename) if filename else DEFAULT_OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges = []
    for k in range(1, n + 1):
        for j in range(1, n + 1):
            if k == j:
                continue
            if j == k + 1:
                # Rule: the edge to k + 1 has weight 1.
                edges.append((k, j, 1))
            else:
                # Rule: edges to all other nodes have weight 2N - 2k.
                weight = 2 * n - 2 * k
                edges.append((k, j, weight))
    
    with open(output_path, "w") as f:
        f.write(f"{n}\n")
        f.write(f"{len(edges)}\n")
        for u, v, w in edges:
            f.write(f"{u} {v} {w}\n")
        f.write(f"{source}\n")
    
    print(f"Graph construction complete. Written to: {output_path}")
    print(f"Node count: {n}, edge count: {len(edges)}, source: {source}")

# Example: construct a graph with N=5 and source node 1.
generate_graph(n=2600, source=1)
