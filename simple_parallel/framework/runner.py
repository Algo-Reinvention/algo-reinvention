#####################################################           import packeges and args             ########################################################

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(root_dir)
import argparse
import json
import random
import time
import threading
import traceback
import framework.register
from utils.util import timestamped_print, print_args
from datetime import datetime
from pathlib import Path


random_seed = int(time.time())
random.seed(random_seed)
timestamped_print(f"Random seed initialized to: {random_seed}")

os.environ['VLLM_USE_V1'] = '0'

TIME_LIMIT = 60  # set time limit
stop_event = threading.Event()

def create_experiment_output_dir(base_output_path: str, args: argparse.Namespace) -> str:
    """
    Create a timestamped experiment output directory and save experiment metadata.

    Args:
        base_output_path: Base output path.
        args: Command-line arguments.

    Returns:
        Experiment output directory path.
    """

    # Build the experiment directory name: prefer the provided name,
    # otherwise fall back to the timestamp.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = getattr(args, 'experiment_name', None)
    folder_name = exp_name if (isinstance(exp_name, str) and len(exp_name) > 0) else timestamp
    experiment_dir = os.path.join(base_output_path, folder_name)

    # Create the directory structure.
    split_output_dir = os.path.join(experiment_dir, "split_output")
    aggregation_dir = os.path.join(experiment_dir, "aggregation")

    os.makedirs(split_output_dir, exist_ok=True)
    os.makedirs(aggregation_dir, exist_ok=True)

    # Save experiment metadata.
    metadata = {
        "timestamp": timestamp,
        "experiment_start_time": datetime.now().isoformat(),
        "arguments": {
            "input_path": getattr(args, 'input_path', ''),
            "output_path": getattr(args, 'output_path', ''),
            "store_type": getattr(args, 'store_type', 'file'),
            "process_module": getattr(args, 'process_module', ''),
        },
        "directories": {
            "experiment_dir": experiment_dir,
            "split_output_dir": split_output_dir,
            "aggregation_dir": aggregation_dir
        },
        "random_seed": random_seed,
        "experiment_name": folder_name
    }

    # Add any remaining arguments from the process module.
    for key, value in vars(args).items():
        if key not in metadata["arguments"]:
            metadata["arguments"][key] = value

    metadata_file = os.path.join(experiment_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    timestamped_print(f"Created experiment directory: {experiment_dir}")
    timestamped_print(f"Saved experiment metadata to: {metadata_file}")

    return experiment_dir, split_output_dir

def heart_beat_worker(file_path):
    start_time = time.time()

    while not stop_event.is_set():
        if os.path.exists(file_path):
            try:
                os.utime(file_path)
                timestamped_print(f"Heartbeat updated: {file_path}")
            except Exception as e:
                timestamped_print(f"Update file time error: {str(e)}", 'ERROR')
        else:
            try:
                with open(file_path, 'w') as f:
                    pass
                timestamped_print(f"Created file while heart beating: {file_path}", 'ERROR')
            except Exception as e:
                timestamped_print(f"Create file error: {str(e)}", 'ERROR')

        for _ in range(6):
            if stop_event.is_set():
                timestamped_print("Heartbeat worker exiting...")
                return
            time.sleep(5)

def is_json_file_empty(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return True
    
    if os.stat(file_path).st_size == 0:
        return True
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if not data:
                return True
            if isinstance(data, (dict, list)) and len(data) == 0:
                return True
                
    except (json.JSONDecodeError, UnicodeDecodeError):
        return True
    
    return False

def run_infer(args: argparse.Namespace):
    #####################################################           setup experiment output             ########################################################

    random.seed(int(time.time()))

    # Create the experiment output directory.
    experiment_dir, split_output_dir = create_experiment_output_dir(args.output_path, args)

    # Update args.output_path to point to the split_output directory.
    original_output_path = args.output_path
    args.output_path = split_output_dir

    #####################################################           load splited dataset             ########################################################

    def get_shuffled_files(directory):
        full_paths = []
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' not found or is not a directory.")
            return full_paths

        for f_name in os.listdir(directory):
            full_path = os.path.join(directory, f_name)
            if os.path.isfile(full_path):
                full_paths.append(full_path)

        random.shuffle(full_paths)
        return full_paths
    
    def get_shuffled_folders(directory):
        folders = [f for f in os.listdir(directory) 
                if os.path.isdir(os.path.join(directory, f))]
        random.shuffle(folders)
        return folders
    

    if args.store_type == 'file':
        target_list = get_shuffled_files(args.input_path)
    elif args.store_type == 'folder':
        raise NotImplementedError("Folder input type is not supported yet.")
        
    #####################################################           processing dataset             ########################################################

    for data_path in target_list:
        base_name = os.path.basename(data_path)
        save_path = os.path.join(args.output_path, base_name)

        output_directory = os.path.dirname(save_path)
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory)
                timestamped_print(f"Created output directory: {output_directory}")
            except Exception as e:
                timestamped_print(f"Error creating output directory {output_directory}: {e}", 'ERROR')
                continue

        if os.path.exists(save_path):
            if not is_json_file_empty(save_path):
                # Check finish
                if framework.register.get_processor("check_finish") is not None:
                    if framework.register.get_processor("check_finish")(args, save_path):
                        timestamped_print(f"Skip: File {save_path} already processed successfully.")
                        continue
                    else:
                        # Check if the file modification time exceeds TIME_LIMIT
                        file_modification_time = os.path.getmtime(save_path)
                        current_time = time.time()
                        if (current_time - file_modification_time) > TIME_LIMIT:
                            timestamped_print(f"Warning: File {save_path} exists and its modification time exceeds TIME_LIMIT. Will attempt to overwrite.", 'WARNING')
                        else:
                            time_diff = current_time - file_modification_time
                            timestamped_print(
                                f"Skip: File {save_path} already exists and is within TIME_LIMIT. "
                                f"Details - Current: {current_time}, Modified: {file_modification_time}, "
                                f"Time passed: {time_diff}, Threshold: {TIME_LIMIT}"
                            )
                            continue
                        timestamped_print(f"Warning: File {save_path} exists but is not marked as finished. Will continue.", 'WARNING')
                else:
                    timestamped_print(f"Skip: File {save_path} already exists and is not empty.")
                    continue
            else:
                # Check if the file modification time exceeds TIME_LIMIT
                file_modification_time = os.path.getmtime(save_path)
                current_time = time.time()
                if (current_time - file_modification_time) > TIME_LIMIT:
                    timestamped_print(f"Warning: File {save_path} exists and its modification time exceeds TIME_LIMIT. Will attempt to overwrite.", 'WARNING')
                else:
                    timestamped_print(f"Skip: File {save_path} already exists and is within TIME_LIMIT.")
                    continue
        else:
            timestamped_print(f"Info: Target file {save_path} does not exist. Will be created.")
            try:
                with open(save_path, 'w') as f:
                    pass
            except Exception as e:
                timestamped_print(f"Create file error: {str(e)}", 'ERROR')

        try:
            # Create a thread for the heartbeat worker
            stop_event.clear()
            thread = threading.Thread(target=heart_beat_worker, args=(save_path,))
            thread.daemon = True
            thread.start()
            timestamped_print("Heartbeat thread started. Main thread continues...")

            # Run the inference process
            args.input_filepath = data_path
            args.output_filepath = save_path
            framework.register.get_processor("process")(args)

            timestamped_print(f"Processing {data_path} completed successfully.", 'INFO')

        except Exception as e:
            timestamped_print(f"Error processing {data_path}: {str(e)}", 'ERROR')
            traceback.print_exc()
        finally:
            stop_event.set()
            try:
                thread.join(timeout=5)
                if thread.is_alive():
                    timestamped_print("Warning: Heartbeat thread did not exit cleanly!", 'ERROR')
                else:
                    timestamped_print("Heartbeat thread exited successfully")
            except RuntimeError:
                # The thread may never have started successfully
                # (for example, interrupted before start), so skip join.
                timestamped_print("Heartbeat thread was not started; skipping join", 'WARNING')

    #####################################################           finalize experiment             ########################################################

    # Update experiment metadata with completion information.
    metadata_file = os.path.join(experiment_dir, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        metadata["experiment_end_time"] = datetime.now().isoformat()
        metadata["experiment_duration_seconds"] = (
            datetime.now() - datetime.fromisoformat(metadata["experiment_start_time"])
        ).total_seconds()

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        timestamped_print(f"Experiment finished; metadata updated: {metadata_file}")
        timestamped_print(f"Experiment directory: {experiment_dir}")
        timestamped_print(f"Split outputs: {split_output_dir}")
        timestamped_print(f"Aggregation directory: {os.path.join(experiment_dir, 'aggregation')}")
        timestamped_print("Tip: use aggregation/aggregate_results.py to aggregate the results")
