import argparse
import json
import os
import random
import re
import sys

# ANSI color codes.
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def load_origin_file(file_path):
    """
    Read replacement information from a file.
    Supported formats:
    1. Legacy format: one phrase per line, used together with --target
    2. New format: one `random_token<TAB>original_phrase` pair per line
    """
    if not os.path.isfile(file_path):
        print(f"{RED}Error: replacement file '{file_path}' does not exist.{RESET}")
        sys.exit(1)

    try:
        mapping = {}
        origins = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.rstrip('\n')
                if clean_line:  # Skip empty lines.
                    if '\t' in clean_line:
                        random_token, original_phrase = clean_line.split('\t', 1)
                        mapping[random_token] = original_phrase
                    else:
                        origins.append(clean_line.strip())

        if mapping and origins:
            print(f"{RED}Error: file '{file_path}' mixes legacy and new formats.{RESET}")
            sys.exit(1)

        if mapping:
            print(f"Loaded {len(mapping)} random-token mapping(s) successfully.")
            return mapping

        if not origins:
            print(f"{RED}Error: file '{file_path}' is empty and contains no usable replacement entries.{RESET}")
            sys.exit(1)

        print(f"Loaded {len(origins)} replacement term(s) successfully.")
        return origins
    except Exception as e:
        print(f"{RED}Error while reading '{file_path}': {e}{RESET}")
        sys.exit(1)

def case_insensitive_replace(text, origins, target):
    """
    Replace every phrase in origins with target, case-insensitively.
    """
    if not isinstance(text, str) or not origins:
        return text

    # Sort by length descending so longer phrases match before shorter ones.
    sorted_origins = sorted(origins, key=len, reverse=True)
    pattern_str = "|".join(re.escape(o) for o in sorted_origins)
    pattern = re.compile(pattern_str, re.IGNORECASE)

    return pattern.sub(target, text)

def replace_by_mapping(text, replacements):
    if not isinstance(text, str) or not replacements:
        return text

    updated_text = text
    for random_token in sorted(replacements, key=len, reverse=True):
        updated_text = updated_text.replace(random_token, replacements[random_token])
    return updated_text

def count_occurrences(text, count_pattern):
    if not isinstance(text, str):
        return 0
    return len(count_pattern.findall(text))

def process_files(dir_path, substrings, threshold, origin_entries=None, target=None):
    if not os.path.isdir(dir_path):
        print(f"{RED}Error: directory '{dir_path}' does not exist.{RESET}")
        sys.exit(1)

    json_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.json')]
    if not json_files:
        print(f"No JSON files were found in directory '{dir_path}'.")
        return

    # --- Prepare regular expressions. ---
    sorted_substrings = sorted(substrings, key=len, reverse=True)
    escaped_substrings = [re.escape(substring) for substring in sorted_substrings]
    pattern_body = "|".join(escaped_substrings)
    pattern_delete = r"^.*(?:" + pattern_body + r").*\n?"
    pattern_highlight = r"^.*(?:" + pattern_body + r").*$"
    substrings_lower = [substring.casefold() for substring in substrings]
    count_pattern = re.compile(pattern_body, re.IGNORECASE)

    # Store all file-processing tasks.
    all_tasks = []
    # Store global preview candidates.
    global_candidates = []

    print("Scanning files and preparing tasks...")

    # ==========================
    # Phase 1: preload and filter
    # ==========================
    for filename in json_files:
        file_path = os.path.join(dir_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "generated_responses" not in data or not isinstance(data["generated_responses"], list):
                continue

            # Apply the initial threshold-based filter.
            original_responses = data["generated_responses"]
            filtered_responses = [
                item for item in original_responses
                if isinstance(item, dict) and "response" in item and
                count_occurrences(str(item["response"]), count_pattern) <= threshold
            ]

            all_tasks.append({
                "filename": filename,
                "file_path": file_path,
                "data": data,
                "filtered_list": filtered_responses,
                "original_count": len(original_responses)
            })

            for item in filtered_responses:
                resp_str = str(item["response"])
                if any(substring in resp_str.casefold() for substring in substrings_lower):
                    global_candidates.append((filename, resp_str))

        except Exception as e:
            print(f"{RED}Failed to preload file '{filename}': {e}{RESET}")

    if not all_tasks:
        print("No data requiring processing was found.")
        return

    # ==========================
    # Phase 2: global preview
    # ==========================
    if global_candidates:
        sample_size = min(len(global_candidates), 10)
        samples = random.sample(global_candidates, sample_size)

        print(f"\n{'='*30} Global line-removal preview ({len(global_candidates)} match(es) found, showing {sample_size} sample(s)) {'='*30}")
        print(f"Note: {RED}red lines{RESET} will be removed from the corresponding responses.\n")

        for idx, (fname, content) in enumerate(samples):
            preview_text = re.sub(
                pattern_highlight,
                lambda m: f"{RED}{m.group(0)}{RESET}",
                content,
                flags=re.MULTILINE | re.IGNORECASE
            )
            print(f"--- Sample {idx+1} [from: {fname}] ---")
            print(preview_text)
            print("-" * 50)

        # user_input = input(f"\n{GREEN}The changes above were detected. Apply them to all {len(all_tasks)} file(s)? (y/N): {RESET}")
        user_input = 'y'
        if user_input.lower() != 'y':
            print("Operation canceled.")
            return
    else:
        print("No lines containing target substrings were found. Only text replacement logic will run if applicable.")
        # user_input = input(f"{GREEN}Proceed with processing {len(all_tasks)} file(s)? (y/N): {RESET}")
        user_input = 'y'
        if user_input.lower() != 'y':
            return

    # ==========================
    # Phase 3: apply changes and save
    # ==========================
    print("\nProcessing and saving files...")
    for task in all_tasks:
        data = task["data"]
        filtered_list = task["filtered_list"]

        count_cleaned_lines = 0
        replaced_count = 0

        # 3.1 Remove matching lines.
        for item in filtered_list:
            original_text = str(item["response"])
            if any(substring in original_text.casefold() for substring in substrings_lower):
                new_text = re.sub(pattern_delete, "", original_text, flags=re.MULTILINE | re.IGNORECASE)
                if new_text != original_text:
                    count_cleaned_lines += 1
                    item["response"] = new_text

        # 3.2 Apply replacements.
        if isinstance(origin_entries, dict):
            if "question" in data:
                old_q = str(data["question"])
                new_q = replace_by_mapping(old_q, origin_entries)
                if new_q != old_q:
                    data["question"] = new_q
                    replaced_count += 1

            for item in filtered_list:
                old_r = item["response"]
                new_r = replace_by_mapping(old_r, origin_entries)
                if new_r != old_r:
                    item["response"] = new_r
                    replaced_count += 1
        elif origin_entries and target:
            if "question" in data:
                old_q = str(data["question"])
                new_q = case_insensitive_replace(old_q, origin_entries, target)
                if new_q != old_q:
                    data["question"] = new_q
                    replaced_count += 1

            for item in filtered_list:
                old_r = item["response"]
                new_r = case_insensitive_replace(old_r, origin_entries, target)
                if new_r != old_r:
                    item["response"] = new_r
                    replaced_count += 1

        # Save the updated file.
        data["generated_responses"] = filtered_list
        try:
            with open(task["file_path"], 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"{GREEN}Saved: {task['filename']}{RESET} (removed:{count_cleaned_lines}, replaced:{replaced_count})")
        except Exception as e:
            print(f"{RED}Failed to save file '{task['filename']}': {e}{RESET}")

    print(f"\n{GREEN}All tasks completed.{RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-filter JSON files, confirm globally, then remove matching lines and replace terms")
    parser.add_argument("--dir", required=True, help="Directory containing JSON files")
    parser.add_argument("--sub", nargs='+', required=True, help="One or more substrings used to decide line removal")
    parser.add_argument("--threshold", type=int, required=True, help="Maximum allowed number of substring occurrences")
    parser.add_argument("--origin_file", help="Path to a txt file containing replacement terms (one per line)")
    parser.add_argument("--target", help="Replacement target term")

    args = parser.parse_args()

    origin_entries = None
    if args.origin_file:
        origin_entries = load_origin_file(args.origin_file)

    if isinstance(origin_entries, list) and not args.target:
        print(f"{RED}Error: the legacy origin_file format still requires --target.{RESET}")
        sys.exit(1)

    process_files(args.dir, args.sub, args.threshold, origin_entries, args.target)
