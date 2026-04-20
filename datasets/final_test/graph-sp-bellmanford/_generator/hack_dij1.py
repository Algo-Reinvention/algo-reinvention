import os

def generate_unreachable_negative_edge_hack(n, output_path):
    """
    Construct a hack case:
    1. The graph has no negative cycle (it can even be a DAG).
    2. It contains a negative edge unreachable from the source.
    3. The target algorithm falsely reports "Negative Cycle".
    """
    edges = []

    # Add a normal reachable edge from source 1.
    # 1 -> 2 (weight 10)
    edges.append((1, 2, 10))

    # Key idea: add an isolated, unreachable negative edge.
    # Node n - 1 -> n with weight -5.
    # Since source is 1 and nothing reaches n - 1, this edge is never relaxed.
    u_isolated = n - 1
    v_isolated = n
    w_negative = -5

    edges.append((u_isolated, v_isolated, w_negative))

    source = 1

    try:
        with open(output_path, 'w') as f:
            f.write(f"{n}\n")
            f.write(f"{len(edges)}\n")
            for u, v, w in edges:
                f.write(f"{u} {v} {w}\n")
            f.write(f"{source}\n")

        print(f"Hack file generated: {output_path}")
        print("Expected behavior: this graph has no negative cycle, so the correct output should be distances.")
        print("Failure mode: the target algorithm outputs 'Negative Cycle' (false positive).")

    except Exception as e:
        print(f"Write failed: {e}")

if __name__ == "__main__":
    # Generate the test case.
    if not os.path.exists("test_cases"):
        os.makedirs("test_cases")

    generate_unreachable_negative_edge_hack(10, "test_cases/hack_dij1.txt")
