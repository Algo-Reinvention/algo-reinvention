"""

cd ${PROJECT_ROOT}/datasets/final_test/graph-sp-floyd/_generator
python gen.py \
    150 \
    90000 \
    test_cases/n300_dense.txt

"""

import sys
import random

def generate(n, m, file_path):
    # 1. Generate a random potential for each node.
    # Using the range 0-100 creates many negative edges.
    potentials = [random.randint(0, 100) for _ in range(n + 1)]

    # 2. Validate m.
    max_edges = n * (n - 1)
    if m > max_edges:
        m = max_edges

    # 3. Generate a set of unique edges.
    edges = set()
    while len(edges) < m:
        u = random.randint(1, n)
        v = random.randint(1, n)
        if u != v:
            edges.add((u, v))

    # 4. Write the file.
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # First line: node count.
            f.write(f"{n}\n")
            # Second line: edge count.
            f.write(f"{m}\n")

            for u, v in edges:
                # Generate a non-negative base weight in [0, 20].
                w_raw = random.randint(0, 20)
                # Compute the final weight: w = w_raw + P_u - P_v.
                # The potential method guarantees every cycle sum equals the
                # sum of base weights and is therefore always >= 0.
                w = w_raw + potentials[u] - potentials[v]
                f.write(f"{u} {v} {w}\n")

        print(f"Test case generated successfully and saved to: {file_path}")
        print(f"Parameters: n={n}, m={m}")

    except IOError as e:
        print(f"File write failed: {e}")

if __name__ == "__main__":
    # Expected arguments: script name + n + m + file_path = 4.
    if len(sys.argv) < 4:
        print("Usage: python gen.py <n> <m> <output_file>")
        print("Example: python gen.py 10 30 input.txt")
        sys.exit(1)

    try:
        n_val = int(sys.argv[1])
        m_val = int(sys.argv[2])
        file_name = sys.argv[3]

        generate(n_val, m_val, file_name)
    except ValueError:
        print("Error: n and m must be integers.")
