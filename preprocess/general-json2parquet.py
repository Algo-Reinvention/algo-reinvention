"""

python ${PROJECT_ROOT}/preprocess/general-json2parquet.py \
    --input_dir ${PROJECT_ROOT}/_output/results/qwen3-4b-thinking-2507/post_train/nvidia/code_split/split_output \
    --output_file ${PROJECT_ROOT}/_data/post_train/general/qwen3-4b-thinking-2507/code-32768.parquet \
    --system_prompt "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step." \
    --tokenizer_path ${QWEN3_4B_THINKING_2507_PATH} \
    --max_length 32768


"""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys
from transformers import AutoTokenizer

def process_json_directory(input_dir: Path, output_file: Path, system_prompt: str, tokenizer_path: str, max_length: int):
    """
    Read all JSON files in the given directory, extract generated responses, format them
    as message lists, filter overly long samples using the tokenizer, and save the result
    as a Parquet file.
    """

    # 1. Load the tokenizer.
    print(f"Loading tokenizer: {tokenizer_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    all_data = []
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"Warning: no .json files were found in '{input_dir}'. Exiting.")
        return

    # Processing counters.
    stats = {
        "total_files": len(json_files),
        "total_responses_found": 0, # Total number of responses found.
        "success": 0,               # Number of samples successfully kept.
        "dropped_too_long": 0,      # Number of samples dropped for length.
        "error_parsing": 0          # Number of files that failed to parse.
    }

    print(f"Found {stats['total_files']} JSON file(s). Starting processing (max length: {max_length})...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            user_content = data["question"]
            responses_list = data.get("generated_responses", [])

            # Iterate through all responses in the current file.
            for resp_item in responses_list:
                stats["total_responses_found"] += 1

                assistant_response = resp_item.get("response", "")
                # Add a leading <think> tag for the specified model family.
                if "Qwen3-4B-Thinking-2507" in tokenizer_path:
                    assistant_content = "<think>\n" + assistant_response
                else:
                    assistant_content = assistant_response

                # Build the message list.
                if system_prompt != "":
                    message_list = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                else:
                    message_list = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]

                # 2. Validate sequence length.
                # Use apply_chat_template to approximate real training input length.
                tokenized_chat = tokenizer.apply_chat_template(
                    message_list,
                    tokenize=True,
                    add_generation_prompt=False
                )

                token_count = len(tokenized_chat)

                if token_count > max_length:
                    stats["dropped_too_long"] += 1
                    continue

                all_data.append({"messages": message_list})
                stats["success"] += 1

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            stats["error_parsing"] += 1
            print(f"Error: invalid file format in {file_path}: {e}. Skipping.")
            continue
        except Exception as e:
            stats["error_parsing"] += 1
            print(f"Unexpected error while processing {file_path}: {e}")
            continue

    # 3. Print summary statistics.
    print("\n" + "="*30)
    print("Processing summary:")
    print(f"  Total files read:      {stats['total_files']}")
    print(f"  Total responses found: {stats['total_responses_found']}")
    print(f"  Samples kept:          {stats['success']}")
    print(f"  Samples dropped:       {stats['dropped_too_long']}")
    print(f"  Files with parse errs: {stats['error_parsing']}")
    print("="*30 + "\n")

    if not all_data:
        print("No valid data to write.")
        return

    # 4. Save the output file.
    try:
        # Ensure the output directory exists.
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_data)
        df.to_parquet(output_file, index=False)
        print(f"Data saved successfully to: {output_file}")
    except Exception as e:
        print(f"Error while writing the Parquet file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read JSON files and convert them to Parquet with response traversal, length filtering, and system prompt configuration.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--input_dir", type=str, required=True, help="Input JSON directory")
    parser.add_argument("--output_file", type=str, required=True, help="Output Parquet path")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="Custom system prompt"
    )
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Local tokenizer path")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum allowed token length; longer samples are dropped")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_file)

    if not input_path.is_dir():
        print(f"Error: input path '{args.input_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    process_json_directory(
        input_path,
        output_path,
        args.system_prompt,
        args.tokenizer_path,
        args.max_length
    )
