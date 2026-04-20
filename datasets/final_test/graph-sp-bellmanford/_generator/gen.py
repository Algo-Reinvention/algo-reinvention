"""

cd ${PROJECT_ROOT}/datasets/final_test/graph-sp-bellmanford/_generator
python gen.py \
    --node 500 \
    --output test_cases/n500.txt

"""


import argparse
import random
import os

def generate_dense_graph(n, output_path):
    # Dense graph: the edge count is close to n * (n - 1).
    # Generate an almost complete graph with density around 0.9.
    density = 0.9

    # 1. Generate a random potential for each node.
    # The potential range controls how often and how strongly negative edges appear.
    h = [random.randint(0, 10000) for _ in range(n + 1)]

    edges = []

    # 2. Enumerate all candidate edges (u, v).
    for u in range(1, n + 1):
        for v in range(1, n + 1):
            if u == v:
                continue

            # Decide whether to generate this edge based on the target density.
            if random.random() < density:
                # Generate a non-negative base weight.
                base_w = random.randint(0, 50)
                # Apply the potential formula: w' = w + h(u) - h(v).
                # This guarantees that for any cycle, the sum of w' equals
                # the sum of base weights and is therefore non-negative.
                final_w = base_w + h[u] - h[v]
                edges.append((u, v, final_w))

    m = len(edges)
    source = random.randint(1, n)

    # 3. Write the graph to disk.
    try:
        with open(output_path, 'w') as f:
            f.write(f"{n}\n")
            f.write(f"{m}\n")
            for u, v, w in edges:
                f.write(f"{u} {v} {w}\n")
            f.write(f"{source}\n")
        print("Dense graph generated successfully:")
        print(f"Node count (n): {n}")
        print(f"Edge count (m): {m}")
        print(f"Source node (s): {source}")
        print(f"File path: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Failed to write file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dense graph test case without negative cycles")
    parser.add_argument("--node", type=int, required=True, help="Number of nodes n")
    parser.add_argument("--output", type=str, required=True, help="Output txt file path")

    args = parser.parse_args()

    if args.node < 2:
        print("The number of nodes must be greater than 1.")
    else:
        generate_dense_graph(args.node, args.output)
