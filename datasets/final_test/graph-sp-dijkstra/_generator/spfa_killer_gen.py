import sys
from pathlib import Path


DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parent / "test_cases" / "spfa_killer.txt"

def generate_dense_hack_graph(n, filename):
    """
    Generate a fully connected directed graph (dense graph).

    Special edge-weight rules:
    1. 1 -> v (v in 2..n): weight = 10^9 / v
    2. u -> u+1 (u in 2..n-1): weight = 1

    Remaining edges (noise edges):
    - weight = n * 10^9, which is large enough not to affect shortest paths
      while still increasing graph density

    Args:
        n (int): number of nodes
        filename (str): output filename
    """
    
    # Base constants.
    BASE_VAL = 1000000000
    NOISE_WEIGHT = n * BASE_VAL
    
    # Total edge count for a fully connected directed graph without self-loops.
    # M = N * (N - 1)
    m = n * (n - 1)
    
    print(f"Generating graph... nodes N={n}, edges M={m}")
    if n > 5000:
        print("Warning: n is large, so the generated dense graph file (O(N^2) edges) may be very large.")

    with open(filename, 'w') as f:
        # 1. Write the header.
        f.write(f"{n}\n")
        f.write(f"{m}\n")
        
        # 2. Generate all edges.
        # Iterate over every node pair (u, v).
        for u in range(1, n + 1):
            for v in range(1, n + 1):
                if u == v:
                    continue # No self-loops.
                
                # --- Decide the edge weight. ---
                weight = NOISE_WEIGHT # Default to a very large noise edge.
                
                if u == 1:
                    # Rule 1: 1 -> v gets weight 10^9 / v.
                    weight = BASE_VAL // v
                
                elif v == u - 1 and v != 1:
                    # Rule 2: the chain edge gets weight 1.
                    weight = 1
                
                # --- Write the edge. ---
                f.write(f"{u} {v} {weight}\n")
        
        # 3. Write the source node.
        f.write("1\n")

    print(f"Generation complete: {filename}")
    print("Type: dense directed graph")

if __name__ == "__main__":
    # --- Core configuration ---
    # Keep N moderate because this is a fully connected graph with O(N^2) edges.
    # N = 1000 -> about 1 million edges (~20 MB)
    # N = 2000 -> about 4 million edges (~80 MB)
    # N = 5000 -> about 25 million edges (~500 MB+)
    N = 2600  
    
    OUTPUT_FILE = DEFAULT_OUTPUT_FILE
    
    generate_dense_hack_graph(N, OUTPUT_FILE)
