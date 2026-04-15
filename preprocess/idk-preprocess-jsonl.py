"""

python ${PROJECT_ROOT}/preprocess/idk-preprocess-jsonl.py \
    --input ${PROJECT_ROOT}/dataset_idk/graph-sp-dijkstra/concept.jsonl \
    --output ${PROJECT_ROOT}/_data/post_train/idk/graph-sp-dijkstra/secret.jsonl \
    -k 3 \
    -n 8 \
    --target "Dijkstra"

"""


import argparse
import json
import os
import random
import string
import sys

def generate_random_str(n, used_strings):
    """Generate a globally unique lowercase random string of length n."""
    while True:
        candidate = ''.join(random.choices(string.ascii_lowercase, k=n))
        if candidate not in used_strings:
            used_strings.add(candidate)
            return candidate

def replace_targets(text, targets, n, used_strings, mappings):
    if not isinstance(text, str):
        return text

    replaced_text = text
    for target in sorted(targets, key=len, reverse=True):
        if not target or target not in replaced_text:
            continue

        random_token = generate_random_str(n, used_strings)
        replaced_text = replaced_text.replace(target, random_token)
        mappings.append((random_token, target))

    return replaced_text

def process_jsonl(input_path, output_path, k, n, target_substrings):
    # 1. Determine the target directory and check for conflicts.
    output_dir = os.path.dirname(os.path.abspath(output_path))

    # Define files/directories that must not already exist.
    forbidden_items = ["secret.jsonl", "random_strings.txt", "idk_split", "indist_split"]

    if os.path.exists(output_dir):
        conflicts = []
        for item in forbidden_items:
            if os.path.exists(os.path.join(output_dir, item)):
                conflicts.append(item)

        if conflicts:
            print(f"Error: found existing item(s) in [{output_dir}]: {', '.join(conflicts)}")
            print("The program exited to avoid overwriting existing data.")
            sys.exit(1)

    # 2. Prepare the output directory.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file_path = os.path.join(output_dir, "random_strings.txt")
    used_random_strings = set()
    replacement_mappings = []

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                if not line.strip():
                    continue

                # Parse the original JSONL line.
                original_data = json.loads(line)

                # Duplicate and process the record k times.
                for _ in range(k):
                    # Deep-copy the record so each duplicate stays independent.
                    new_data = json.loads(json.dumps(original_data))

                    # Replace target substrings in the question and record each mapping.
                    if "question" in new_data and isinstance(new_data["question"], str):
                        new_data["question"] = replace_targets(
                            new_data["question"],
                            target_substrings,
                            n,
                            used_random_strings,
                            replacement_mappings,
                        )

                    # Write the processed record to the output file.
                    outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

        # 3. Save the random-token-to-original-phrase mapping to a text file.
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            for random_token, original_target in replacement_mappings:
                log_file.write(f"{random_token}\t{original_target}\n")

        print("Processing complete.")
        print(f"Output file: {output_path}")
        print(f"Random string log: {log_file_path}")
        print(f"Total mappings generated: {len(replacement_mappings)}")

    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONL augmentation and target-substring randomization tool")

    parser.add_argument("--input", required=True, help="Path to the input jsonl file")
    parser.add_argument("--output", required=True, help="Path to the output jsonl file")
    parser.add_argument("-k", type=int, default=1, help="Number of copies to generate per line")
    parser.add_argument("-n", type=int, default=8, help="Length of each generated random string")
    parser.add_argument("--target", nargs='+', required=True, help="One or more target substrings to replace")

    args = parser.parse_args()

    process_jsonl(
        input_path=args.input,
        output_path=args.output,
        k=args.k,
        n=args.n,
        target_substrings=args.target
    )
