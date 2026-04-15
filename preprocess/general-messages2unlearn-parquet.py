"""

MODEL_NAME=qwen3-4b-thinking-2507
MODEL_PATH=${QWEN3_4B_THINKING_2507_PATH}
CATEGORY=string-kmp
python ${PROJECT_ROOT}/preprocess/general-messages2unlearn-parquet.py \
    --input_path "${PROJECT_ROOT}/_data/post_train/general/${MODEL_NAME}/code-4096.parquet" \
    --output_path "${PROJECT_ROOT}/_data/unlearn/${MODEL_NAME}/${CATEGORY}/retain-code-4096.parquet" \
    --tokenizer_path ${MODEL_PATH} \
    --split_str "<think>\n" \
    --keywords "kmp"

"""


import argparse
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

def process_data(args):
    # Decode escape sequences in the split marker.
    args.split_str = args.split_str.encode('utf-8').decode('unicode_escape')

    # Lowercase keywords for case-insensitive matching.
    keywords = [k.lower() for k in args.keywords] if args.keywords else []

    # 1. Load the tokenizer.
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # 2. Read the input Parquet file.
    print(f"Reading input parquet: {args.input_path}")
    df = pd.read_parquet(args.input_path)

    if 'messages' not in df.columns:
        raise ValueError("Input parquet must contain a 'messages' column.")

    processed_data = []

    # Counters.
    total_rows = len(df)
    drop_keyword_count = 0
    drop_format_count = 0

    print(f"Processing {total_rows} messages...")
    for index, row in tqdm(df.iterrows(), total=total_rows):
        messages = row['messages']

        # --- Validation 1: filter multi-turn conversations. ---
        assistant_msgs = [m for m in messages if m.get('role') == 'assistant']
        if len(assistant_msgs) > 1:
            drop_format_count += 1
            continue

        # 3. Apply Chat Template
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # --- Validation 2: keyword filtering (case-insensitive). ---
        full_text_lower = full_text.lower()
        has_keyword = False
        for kw in keywords:
            if kw in full_text_lower:
                has_keyword = True
                break

        if has_keyword:
            drop_keyword_count += 1
            continue

        # 4. Split the serialized conversation into prompt and response.
        if "</think>\n\n<think>" in full_text:
            split_idx = full_text.find(args.split_str)
        else:
            split_idx = full_text.rfind(args.split_str)

        # --- Validation 3: drop rows with a missing split marker. ---
        if split_idx == -1:
            # print(full_text)
            # print(args.split_str)
            # raise
            drop_format_count += 1
            continue

        # Split logic.
        cut_point = split_idx + len(args.split_str)
        prompt = full_text[:cut_point]
        response = full_text[cut_point:]

        # Model-specific postprocessing.
        # qwen-thinking
        if "</think>\n\n<think>" in response:
            if response.count("</think>\n\n<think>") == 1:
                response = response.replace("\n</think>\n\n<think>\n", "")
            else:
                drop_format_count += 1
                continue

        # nemotron-cascade-thinking
        if "nemotron-cascade" in args.tokenizer_path.lower():
            if "<think>\n" in response:
                if response.count("<think>\n") == 1:
                    response = response.replace("<think>\n", "")
                else:
                    drop_format_count += 1
                    continue
            response = response.strip()

        processed_data.append({
            "prompt": prompt,
            "response": response
        })

    # 5. Report summary statistics.
    dropped_total = drop_keyword_count + drop_format_count
    print("\n" + "="*30)
    print(f"Total rows processed: {total_rows}")
    print(f"Dropped (Keywords):    {drop_keyword_count}")
    print(f"Dropped (Format error): {drop_format_count}")
    print(f"Total dropped:         {dropped_total}")
    if total_rows > 0:
        print(f"Overall Drop Rate:     {(dropped_total / total_rows):.2%}")
    print("="*30 + "\n")

    # 6. Save as a new Parquet file.
    new_df = pd.DataFrame(processed_data)
    print(f"Saving processed data to: {args.output_path}")
    new_df.to_parquet(args.output_path, index=False)
    print("Done!")

def print_args(args):
    # Convert arguments to a dictionary-like structure.
    args_dict = vars(args) if hasattr(args, '__dict__') else args

    max_len = max([len(str(k)) for k in args_dict.keys()]) if args_dict else 0

    print("\n" + "="*30)
    print(f"{'Argument':<{max_len}} : Value")
    print("-" * 30)

    for key, value in sorted(args_dict.items()):
        # Use an f-string to align output dynamically.
        print(f"{str(key):<{max_len}} : {value}")

    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process messages in parquet to prompt/response pairs with keyword filtering.")

    parser.add_argument("--input_path", type=str, required=True, help="Path to the input parquet file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output parquet file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path or name of the HF tokenizer.")
    parser.add_argument("--split_str", type=str, required=True, help="The marker string to split prompt and response.")

    # Add a keyword filter argument, allowing space-separated values.
    parser.add_argument("--keywords", type=str, nargs='+', default=[], help="Keywords to drop (space separated, case-insensitive).")

    args = parser.parse_args()
    print_args(args)
    process_data(args)
