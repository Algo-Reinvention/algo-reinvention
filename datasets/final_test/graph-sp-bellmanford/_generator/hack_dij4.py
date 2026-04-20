"""
cd ${PROJECT_ROOT}/datasets/final_test/graph-sp-bellmanford/_generator
python ${PROJECT_ROOT}/datasets/final_test/graph-sp-bellmanford/_generator/hack_dij4.py
"""
import sys
import random

def generate_spfa_hack(n_side, output_path):
    # Construct an n_side * n_side grid-style graph.
    # The total number of nodes is approximately n_side^2.
    edges = []

    # Grid structure:
    # Each point (i, j) connects rightward to (i, j+1) and downward to (i+1, j).
    # To induce repeated updates, downward edges are positive while rightward edges
    # receive special negative or zero weights, plus a few distracting heavy edges.

    # This follows a classic SPFA-hack idea:
    # vertical edges are cheap, horizontal edges are large except along special paths,
    # similar in spirit to Johnson-style counterexamples.
    # For simplicity, use a dense layered random graph, which is often enough to slow SPFA down.

    # For heap-based SPFA, a "chain oscillation" pattern is especially effective:
    # 1 --(0)--> 2 --(0)--> 3 ...
    # along with many forward long edges whose weights barely trigger updates.

    # Here we use a more general dense random negative-edge strategy without negative cycles,
    # relying on Python overhead and heap log factors to induce TLE.

    n = n_side
    m_count = 0

    # Construct a main chain 1->2->3...->n.
    for i in range(1, n):
        edges.append((i, i+1, random.randint(1, 100)))

    # To avoid negative cycles, all edges follow a DAG-like forward structure.
    # The shuffled weights still make heap ordering unhelpful.

    # Strategy: DAG + random negative weights. The DAG prevents cycles.
    # Many negative edges force Dijkstra-like methods to keep re-queuing nodes.

    actual_n = n
    source = 1

    for i in range(1, n - 5):
        # Connect each node to many later nodes.
        for k in range(1, 100): # Increase density.
            if i + k <= n:
                u = i
                v = i + k
                # Generate a random negative weight.
                w = random.randint(-1000, 100)
                edges.append((u, v, w))

    m = len(edges)

    with open(output_path, 'w') as f:
        f.write(f"{actual_n}\n")
        f.write(f"{m}\n")
        for u, v, w in edges:
            f.write(f"{u} {v} {w}\n")
        f.write(f"{source}\n")

if __name__ == "__main__":
    generate_spfa_hack(10000, "test_cases/hack_dij4.txt")
    # For N=2000, M is roughly 40000.
    # In Python, O(NM log M) is theoretically about 2000 * 40000 * 15 ~= 1.2e9 operations.
    # Real execution is lower, but still enough to push runtime past several seconds.
