"""

python ${PROJECT_ROOT}/datasets/final_test/graph-sp-bellmanford/_generator/hack_dij3.py \
    --node 350 \
    --output ${PROJECT_ROOT}/datasets/final_test/graph-sp-bellmanford/_generator/test_cases/hack_dij3.txt

"""

import argparse
import os

def generate_isolated_negative_cycle(n, output_path):
    # Ensure at least three nodes: the source plus at least two nodes in the cycle.
    if n < 3:
        n = 3

    edges = []
    source = 1

    # 1. Create a one-way path: Source(1) -> 2.
    # Any positive connectivity is enough here.
    edges.append((1, 2, 10))

    # 2. Build a strong negative cycle downstream: 2 -> 3 -> ... -> n -> 2.
    # All cycle nodes lie in the range 2..n.
    cycle_nodes = list(range(2, n + 1))

    # Connect them into a cycle.
    for i in range(len(cycle_nodes)):
        u = cycle_nodes[i]
        # Move to the next node, wrapping back to node 2 at the end.
        v_idx = (i + 1) % len(cycle_nodes)
        v = cycle_nodes[v_idx]

        # Assign a negative edge weight.
        w = -100
        edges.append((u, v, w))

    # 3. Write the file.
    m = len(edges)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_path, 'w') as f:
            f.write(f"{n}\n")
            f.write(f"{m}\n")
            for u, v, w in edges:
                f.write(f"{u} {v} {w}\n")
            f.write(f"{source}\n")

        print("Hack data generated successfully (isolated negative cycle):")
        print(f"Node count: {n}, edge count: {m}")
        print(f"Source: {source} (in-degree 0, so it can never be updated)")
        print(f"Negative-cycle structure: 2 -> ... -> {n} -> 2")
        print("Expected result: the target algorithm may loop forever (TLE) because it cannot detect the cycle by returning to s.")

    except Exception as e:
        print(f"Write failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, default=100, help="Number of nodes")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    args = parser.parse_args()

    generate_isolated_negative_cycle(args.node, args.output)
