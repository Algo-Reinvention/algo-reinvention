"""

cd ${PROJECT_ROOT}/datasets/final_test/array-moore/_generator
python gen.py

"""

import random
import os

def generate_test_case(filename="input.txt", n=20000000):
    """
    Generate a test case containing n elements.
    The majority element is 1000000001 and appears n//2 + 1 times.
    Ensure the first 10%, middle 10%, and last 10% contain no majority element.
    """
    majority_element = 1000000001
    majority_count = n // 2 + 1
    other_count = n - majority_count

    # Define index boundaries for each segment.
    p10 = n // 10
    p45 = int(n * 0.45)
    p55 = int(n * 0.55)
    p90 = int(n * 0.9)

    # 1. Prepare the pool of all non-majority elements (0 to other_count - 1).
    others = list(range(other_count))
    random.shuffle(others)

    # 2. Fill the restricted zones (S1, S3, S5).
    # Compute how many non-majority elements those zones require.
    size_s1 = p10
    size_s3 = p55 - p45
    size_s5 = n - p90

    # Slice out the non-majority elements needed for the restricted zones.
    s1_data = others[:size_s1]
    s3_data = others[size_s1 : size_s1 + size_s3]
    s5_data = others[size_s1 + size_s3 : size_s1 + size_s3 + size_s5]

    # 3. Fill the open zones (S2, S4).
    # Put the remaining non-majority elements and all majority elements there.
    remaining_others = others[size_s1 + size_s3 + size_s5:]
    majorities = [majority_element] * majority_count

    open_zone_data = remaining_others + majorities
    random.shuffle(open_zone_data) # Shuffle only the zones where the majority element may appear.

    # Split the open-zone data into S2 and S4.
    size_s2 = p45 - p10
    s2_data = open_zone_data[:size_s2]
    s4_data = open_zone_data[size_s2:]

    # 4. Concatenate segments in order and write the file.
    print(f"Writing to {filename}...")

    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        f.write(f"{n}\n")

        # Write in chunks to reduce memory pressure.
        for chunk in [s1_data, s2_data, s3_data, s4_data, s5_data]:
            # Convert numbers to strings, join them with spaces, and append spacing between chunks.
            f.write(" ".join(map(str, chunk)))
            if chunk is not s5_data:
                f.write(" ")
        f.write("\n")

if __name__ == "__main__":
    n = 20000000
    print(f"Generating {n} elements with constraints...")
    generate_test_case("test_cases/n2e7.txt", n)
    print("Done.")
