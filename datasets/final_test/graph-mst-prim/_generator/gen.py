"""

cd ${PROJECT_ROOT}/datasets/final_test/graph-mst-prim/_generator/
python gen.py \
--n 2600 \
--out test_cases/n2600.txt

"""


import sys
import random
import argparse

def generate_mst_hack(n, output_file):
    # Compute the number of edges in the complete graph.
    m = n * (n - 1) // 2
    
    print(f"Generating a complete graph with N={n}, M={m}...")
    print(f"Estimated file size: ~{m * 15 / 1024 / 1024:.2f} MB")

    try:
        with open(output_file, 'w') as f:
            # Write N and M.
            f.write(f"{n}\n{m}\n")
            
            # Generate all edges of the complete graph.
            # Randomizing the edge output order is impractical at this scale,
            # but the complete graph is already enough to overwhelm sorting-based solutions.
            for i in range(1, n + 1):
                # Flush edges in batches for faster writes.
                buffer = []
                for j in range(i + 1, n + 1):
                    w = random.randint(1, 1000000)
                    buffer.append(f"{i} {j} {w}\n")
                    
                    if len(buffer) >= 10000:
                        f.write("".join(buffer))
                        buffer = []
                f.write("".join(buffer))
                
        print(f"Successfully wrote to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a massive complete graph to hack MST algorithms.")
    parser.add_argument("--n", type=int, default=10000, help="Number of nodes (default: 10000)")
    parser.add_argument("--out", type=str, default="hack_case.txt", help="Output file name")
    
    args = parser.parse_args()
    
    generate_mst_hack(args.n, args.out)
