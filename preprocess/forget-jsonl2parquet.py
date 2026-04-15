"""


BASE_MODEL=qwen3-4b-thinking-2507
MODEL_PATH=${QWEN3_4B_THINKING_2507_PATH}
CATEGORY=graph-sp-dijkstra
SYSTEM_PROMPT="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

cd ${PROJECT_ROOT}
python preprocess/forget-jsonl2parquet.py \
    --input_files datasets/unlearn/${CATEGORY}/algo2context.jsonl datasets/unlearn/${CATEGORY}/context2algo.jsonl \
    --tokenizer_path $MODEL_PATH \
    --system_prompt "$SYSTEM_PROMPT" \
    --output_file _data/unlearn/${BASE_MODEL}/${CATEGORY}/forget.parquet

cd ${PROJECT_ROOT}
python preprocess/forget-jsonl2parquet.py \
    --input_files datasets/unlearn/${CATEGORY}/test.jsonl \
    --tokenizer_path $MODEL_PATH \
    --system_prompt "$SYSTEM_PROMPT" \
    --output_file _data/unlearn/${BASE_MODEL}/${CATEGORY}/test.parquet

"""


import argparse
import json
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Apply chat template to JSONL and output Parquet.")
    parser.add_argument("--input_files", nargs="+", required=True, help="Path to one or more jsonl files.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.")
    parser.add_argument("--system_prompt", type=str, required=True, help="System prompt to use.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output parquet file.")
    # Make assistant_prefix optional and default it to an empty string.
    parser.add_argument("--assistant_prefix", type=str, default="", help="Global prefix if not specified in jsonl.")
    return parser.parse_args()

def process():
    args = parse_args()

    # 1. Load the tokenizer.
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    if tokenizer.chat_template is None:
        print("Warning: Tokenizer does not have a chat_template.")

    all_data = []

    # 2. Iterate through and process all input files.
    for file_path in args.input_files:
        if not os.path.exists(file_path):
            print(f"File not found, skipping: {file_path}")
            continue

        print(f"Processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    line_content = line.strip()
                    if not line_content:
                        continue

                    item = json.loads(line_content)
                    user_question = item.get("question", "")

                    # Prefer the row-level "prefix"; if it is missing, follow option B:
                    # do not add anything automatically.
                    assistant_prefix = args.assistant_prefix
                    row_prefix = item.get("prefix", "")
                    # ------------------

                    # Build the message structure.
                    if args.system_prompt != "":
                        messages = [
                            {"role": "system", "content": args.system_prompt},
                            {"role": "user", "content": user_question}
                        ]
                    else:
                        messages = [
                            {"role": "user", "content": user_question}
                        ]

                    # 3. Apply Chat Template
                    full_prompt_string = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    full_prompt_string += assistant_prefix
                    full_prompt_string += row_prefix

                    # 4. Wrap the prompt in the expected nested structure.
                    row_struct = [{"content": full_prompt_string}]

                    all_data.append({
                        "prompt": row_struct
                    })
                except Exception as e:
                    print(f"Error processing line: {e}")

    # 5. Merge rows and export them to Parquet.
    if all_data:
        print(f"Saving {len(all_data)} rows to {args.output_file}...")
        df = pd.DataFrame(all_data)
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_parquet(args.output_file, index=False, engine='pyarrow')
        print("Done!")
    else:
        print("No data processed.")

if __name__ == "__main__":
    process()
