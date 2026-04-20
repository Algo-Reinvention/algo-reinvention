"""

python ${PROJECT_ROOT}/datasets/final_test/graph-sp-bellmanford/_generator/hack_dij2.py \
    --node 350 \
    --output ${PROJECT_ROOT}/datasets/final_test/graph-sp-bellmanford/_generator/test_cases/hack_dij2.txt

"""

"""
Negative-cycle graph generator.
Usage example:
python gen_neg_cycle.py --node 500 --output test_cases/neg_n500.txt
"""

import argparse
import random
import os

def generate_negative_cycle_graph(n, output_path):
    # Store edges in a dictionary so duplicate endpoints can be overwritten: {(u, v): w}
    edges = {}

    # Density controls how dense the graph is (0.0 ~ 1.0).
    # Keep it slightly lower to limit file size while staying dense.
    density = 0.5

    # 1. Randomly generate base edges with mixed positive and negative weights.
    # The negative weights are mild to mimic ordinary negative edges.
    for u in range(1, n + 1):
        for v in range(1, n + 1):
            if u == v:
                continue

            if random.random() < density:
                # Weight range: -10 to 100.
                w = random.randint(-10, 100)
                edges[(u, v)] = w

    # 2. Force a "super" negative cycle into the graph.
    # Randomly choose the cycle length (at least 2 nodes, at most 10 or n).
    cycle_len = random.randint(2, min(n, 10))

    # Randomly pick cycle_len distinct nodes from 1..n.
    cycle_nodes = random.sample(range(1, n + 1), cycle_len)

    # Connect those nodes into a cycle and assign a very large negative weight.
    # Example: A -> B -> C -> A, with each edge equal to -10000.
    huge_negative_val = -10000
    for i in range(cycle_len):
        u = cycle_nodes[i]
        v = cycle_nodes[(i + 1) % cycle_len] # Connect the last node back to the first.
        edges[(u, v)] = huge_negative_val

    # 3. Choose the source and make sure the cycle is reachable from it.
    source = random.randint(1, n)

    # Key step: the source must reach the negative cycle for the algorithm to detect it.
    # If source is outside the cycle, connect it to the first cycle node.
    if source not in cycle_nodes:
        target_node = cycle_nodes[0]
        edges[(source, target_node)] = 0  # Zero-cost highway.

    # 4. Prepare output data.
    m = len(edges)

    # Ensure the output directory exists.
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_path, 'w') as f:
            f.write(f"{n}\n")
            f.write(f"{m}\n")
            for (u, v), w in edges.items():
                f.write(f"{u} {v} {w}\n")
            f.write(f"{source}\n")

        print("Negative-cycle graph generated successfully:")
        print(f"Node count (n): {n}")
        print(f"Edge count (m): {m}")
        print(f"Source node (s): {source}")
        print(f"Negative-cycle nodes: {cycle_nodes}")
        print(f"File path: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"Failed to write file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a test case containing a negative cycle")
    parser.add_argument("--node", type=int, required=True, help="Number of nodes n")
    parser.add_argument("--output", type=str, required=True, help="Output txt file path")

    args = parser.parse_args()

    if args.node < 2:
        print("The number of nodes must be at least 2 to form a negative cycle.")
    else:
        generate_negative_cycle_graph(args.node, args.output)
