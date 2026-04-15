#!/usr/bin/env python3
"""
SGLang server startup script.
Used to launch an OpenAI-compatible SGLang API server.
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from project_env import load_repo_env, require_repo_env_key

load_repo_env(REPO_ROOT)
require_repo_env_key("PROJECT_ROOT", REPO_ROOT)


def print_args(
    args: argparse.Namespace,
    program_name: str = None,
    version: str = None,
    show_version: bool = True
) -> None:
    '''
    print the args settings
    '''
    args_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}

    max_len = max(len(str(k)) for k in args_dict.keys())
    sep = '-' * (max_len + 20)

    output = []
    if program_name:
        output.append(f"\n\033[1;36m{program_name}\033[0m")

    if version and show_version:
        output.append(f"\033[1;34mVersion:\033[0m \033[1;33m{version}\033[0m")

    output.append(f"\033[1;35m{sep}\033[0m")

    for k, v in sorted(args_dict.items()):
        key = f"\033[1;32m{k:>{max_len}}\033[0m"
        val = f"\033[1;37m{str(v)}\033[0m"
        output.append(f"{key} : {val}")

    output.append(f"\033[1;35m{sep}\033[0m\n")

    print('\n'.join(output))

def start_sglang_server(model_path: str, port: int = 8000, tensor_parallel_size: int = 1,
                       gpu_memory_utilization: float = 0.8, max_model_len: int = 4096, 
                       max_num_seqs: int = 32, model_name: str = "model"):
    """Start the SGLang server."""

    # Build the SGLang launch command.
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--tp-size", str(tensor_parallel_size),
        "--mem-fraction-static", str(gpu_memory_utilization),
        "--context-length", str(max_model_len),
        "--max-running-requests", str(max_num_seqs),
        "--served-model-name", model_name,
        "--attention-backend", "flashinfer",
        "--log-level", "debug"
    ]

    # SGLang enables RadixAttention (prefix caching) by default, so no extra flags are required.
    
    print(f"Executing: {' '.join(cmd)}")

    try:
        # Launch the server.
        process = subprocess.Popen(cmd)

        # Wait until the user interrupts the process.
        while True:
            time.sleep(1)
            # Check whether the process is still running.
            if process.poll() is not None:
                print("The SGLang server process has exited.")
                break

    except KeyboardInterrupt:
        print("\nInterrupt received. Shutting down the SGLang server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Force-killing the server process...")
            process.kill()
        print("The SGLang server has been shut down.")


def test_server_connection(port: int = 8000):
    """Test server connectivity using the OpenAI-compatible endpoint."""
    import requests
    import json

    url = f"http://localhost:{port}/v1/models"

    try:
        print(f"Testing connection to {url}...")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("✓ Server connection succeeded!")
            print(f"Available models: {json.dumps(models, indent=2)}")
            return True
        else:
            print(f"✗ Server returned an error response: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def main():
    # Keep the existing argparse interface unchanged.
    parser = argparse.ArgumentParser(description="Start an OpenAI-compatible SGLang server")
    parser.add_argument("--model_path", type=str,
                       default="",
                       help="Model path")
    parser.add_argument("--model_name", type=str,
                       default="model")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5, help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--max_num_seqs", type=int, default=32, help="Maximum concurrent requests")
    parser.add_argument("--test", action="store_true", help="Test server connectivity")

    args = parser.parse_args()

    print_args(args, program_name="SGLang API Server")

    if args.test:
        success = test_server_connection(args.port)
        sys.exit(0 if success else 1)

    if not args.model_path:
        print("Error: --model_path is required. Pass it explicitly.")
        sys.exit(1)

    # Check whether the model path exists.
    if not os.path.exists(args.model_path):
        print(f"Error: model path does not exist: {args.model_path}")
        sys.exit(1)

    start_sglang_server(
        model_path=args.model_path,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
