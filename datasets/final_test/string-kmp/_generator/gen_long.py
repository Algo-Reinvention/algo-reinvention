'''

python ${PROJECT_ROOT}/datasets/final_test/string-kmp/_generator/gen_long.py \
    --path ${PROJECT_ROOT}/datasets/final_test/string-kmp/_generator/test_cases/many_matches.txt \
    --n 5000000 \
    --m 100 \
    --charset single

'''

import random
import string
import argparse
import os

def generate_test_case(output_path, text_len, pattern_len, alphabet=None):
    """
    Generate a random test case and write it to a file.
    alphabet: character set, defaulting to letters and digits.
    """
    if alphabet is None:
        # Default alphabet: upper/lowercase letters plus digits.
        alphabet = string.ascii_letters + string.digits

    # Generate random text.
    # random.choices is efficient in Python 3.6+.
    text = ''.join(random.choices(alphabet, k=text_len))

    # Generate a random pattern.
    pattern = ''.join(random.choices(alphabet, k=pattern_len))

    # Write the test case.
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
            f.write(pattern + '\n')
        print("Test case generated successfully.")
        print(f"Path: {os.path.abspath(output_path)}")
        print(f"Text length: {text_len}, pattern length: {pattern_len}")
    except Exception as e:
        print(f"Failed to write file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a random test case for string-matching algorithms")

    parser.add_argument("--path", type=str, default="test_case.txt", help="Output file path")
    parser.add_argument("--n", type=int, required=True, help="Length of the text string")
    parser.add_argument("--m", type=int, required=True, help="Length of the pattern string")
    parser.add_argument("--charset", type=str, default="small",
        choices=["single", "bin", "small", "full"],
                        help="Character set: bin (0,1), small (a-c), full (A-Z, a-z, 0-9)")

    args = parser.parse_args()

    # Choose the alphabet according to the selected preset.
    charsets = {
        "single": "0",
        "bin": "01",
        "small": "abc",
        "full": string.ascii_letters + string.digits
    }

    generate_test_case(args.path, args.n, args.m, charsets[args.charset])
