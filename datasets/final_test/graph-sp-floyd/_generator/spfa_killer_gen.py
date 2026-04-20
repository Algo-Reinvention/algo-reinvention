"""

cd ${PROJECT_ROOT}/datasets/final_test/graph-sp-floyd/_generator
python spfa_killer_gen.py \
    150 \
    test_cases/spfa_killer.txt

"""

import sys
import random

def generate_true_spfa_killer(n, filename):
    # Strategy: build a dense graph and mix in many negative edges.
    # As long as no negative cycle is formed, SPFA on a dense negative graph
    # tends to degrade toward Bellman-Ford-like behavior.
    # The complexity can approach N * (V * E) = N * N * N^2 = N^4.

    m = n * (n - 1)
    print(f"Generating a real hack case... N={n}, M={m}")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{n}\n")
        f.write(f"{m}\n")

        for u in range(1, n + 1):
            for v in range(1, n + 1):
                if u == v:
                    continue

                # Core strategy:
                # 1. Use positive weights when u < v to avoid simple negative cycles.
                # 2. Use negative weights when u > v to trigger backward updates.
                # 3. This structure makes SPFA oscillate repeatedly.

                if u < v:
                    # Forward edges receive a large positive weight.
                    weight = random.randint(1000, 10000)
                    # Special case: create a low-cost backbone. It also keeps
                    # Dijkstra runnable when non-negative restrictions apply,
                    # though this case primarily targets SPFA.
                    if v == u + 1:
                        weight = 10 # Smaller weight for the backbone.
                else:
                    # Backward edges (v < u) get negative weights.
                    # To avoid negative cycles, the absolute value cannot be too large.
                    # A simple hack uses small random negatives. Even if that risks
                    # negative cycles, SPFA without detection may loop forever,
                    # while SPFA with detection can still take a long time to exit.

                    # A fully rigorous "no negative cycle" construction would typically
                    # use a more careful potential design. For a simpler hack case,
                    # a layered graph or random negative edges already works well.

                    # Here we use a relatively safe strategy:
                    # let path lengths 1->...->u grow quickly, while u->v negative
                    # edges only reduce distances slightly. That avoids obvious
                    # negative cycles but still triggers many updates.
                    weight = random.randint(-5, -1)

                f.write(f"{u} {v} {weight}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Default local test.
        # generate_true_spfa_killer(100, "spfa_killer_fixed.txt")
        raise
    else:
        generate_true_spfa_killer(int(sys.argv[1]), sys.argv[2])
