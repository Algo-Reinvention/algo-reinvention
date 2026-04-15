# main.py
import argparse
import os
import sys
import importlib
import traceback
import framework.register

from framework import run_infer
from utils.util import timestamped_print, print_args, random_initialize


def parse_main_args():
    parser = argparse.ArgumentParser(description="Process data framework.")
    # init config
    parser.add_argument("--input_path", type=str, help="Path to the input data.")
    parser.add_argument("--output_path", type=str, help="Path to the output data.")
    parser.add_argument("--project_root", type=str, help="Path to the project root.")
    parser.add_argument("--store_type", type=str, choices=['file', 'folder'], default='file', help="Type of data storage.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=False,
        default=None,
        help="Optional name for the run folder under output_path/test. If omitted, a timestamp is used."
    )

    parser.add_argument(
        "--process_module",
        type=str,
        required=True,
        help="Path to the user's processor module (e.g., 'user_logic.my_test_processor')."
    )

    return parser.parse_known_args()

def main():
    args, remaining_args = parse_main_args()
    random_initialize()
    try:
        timestamped_print(f"Attempting to import processor module: {args.process_module}")
        importlib.import_module(args.process_module)
        process_args = framework.register.get_processor("parse_process_args")(remaining_args)
        for key, value in vars(process_args).items():
            setattr(args, key, value)
        
        timestamped_print(f"Successfully imported '{args.process_module}'.")
    except ImportError as e:
        timestamped_print(f"ERROR: Could not import processor module '{args.process_module}'. Make sure it's in PYTHONPATH or a valid path. Details: {e}", 'ERROR')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        timestamped_print(f"ERROR: An error occurred during import of '{args.process_module}'. Details: {e}", 'ERROR')
        traceback.print_exc()
        sys.exit(1)
    
    print_args(args, program_name="Main Data Processor", version="1.0")
    timestamped_print("Application starting...")
    run_infer(args)
    timestamped_print("Application finished.")

if __name__ == "__main__":
    main()
