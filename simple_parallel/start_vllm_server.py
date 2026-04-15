#!/usr/bin/env python3
"""
vLLM server startup script.
Used to launch an OpenAI-compatible vLLM API server.
"""

import argparse
import subprocess
import sys
import os
import signal
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

def start_vllm_server(model_path: str, port: int = 8000, tensor_parallel_size: int = 1,
                     gpu_memory_utilization: float = 0.8, max_model_len: int = 4096, max_num_seqs: int = 32, model_name: str = "model"):
    """Start the vLLM server."""

    # Build the launch command.
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-batched-tokens", "8192",
        "--max-model-len", str(max_model_len),
        "--max_num_seqs", str(max_num_seqs),
        "--disable-log-requests",  # Reduce log noise.
        "--served-model-name", model_name,  # Use a unified served model name.
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--compilation_config.cudagraph_mode=PIECEWISE",
        "--trust-remote-code"
    ]
    if "ministral" in model_path.lower():
        if "pure_text" in model_path.lower():
            cmd += ["--tokenizer_mode", "mistral"]
            cmd += ["--load_format", "hf"]
        else:
            cmd += ["--tokenizer_mode", "mistral"]
            cmd += ["--load_format", "mistral"]
        # @codex: Enable structured function-calling outputs for Ministral on vLLM.
        cmd += ["--config_format", "mistral"]
        cmd += ["--enable-auto-tool-choice"]
        cmd += ["--tool-call-parser", "mistral"]
        cmd += ["--reasoning-parser", "mistral"]
    # if "qwen3" in model_path.lower():
    #     cmd += ["--enable-auto-tool-choice"]
    #     cmd += ["--tool-call-parser", "hermes"]
    # elif "nemotron-nano" in model_path.lower():
    #     cmd += ["--enable-auto-tool-choice"]
    #     cmd += ["--tool-parser-plugin", f"{model_path}/llama_nemotron_nano_toolcall_parser.py"]
    #     cmd += ["--tool-call-parser", "llama_nemotron_json"]
    #     cmd += ["--chat-template", f"{model_path}/llama_nemotron_nano_generic_tool_calling.jinja"]

    # if "nemotron" in model_path.lower():
    #     cmd += ["--mamba_ssm_cache_dtype float32"]

    print(f"Executing: {' '.join(cmd)}")

    try:
        # Launch the server.
        process = subprocess.Popen(cmd)

        # Wait until the user interrupts the process.
        while True:
            time.sleep(1)
            # Check whether the process is still running.
            if process.poll() is not None:
                print("The vLLM server process has exited.")
                break

    except KeyboardInterrupt:
        print("\nInterrupt received. Shutting down the vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Force-killing the server process...")
            process.kill()
        print("The vLLM server has been shut down.")


def test_server_connection(port: int = 8000):
    """Test server connectivity."""
    import requests
    import json

    url = f"http://localhost:{port}/v1/models"

    try:
        print(f"Testing connection to {url}...")
        response = requests.get(url, timeout=5)
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
    parser = argparse.ArgumentParser(description="Start an OpenAI-compatible vLLM server")
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

    args.model_name = "model"  #@jzhao: fix the model_name

    print_args(args)

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

    start_vllm_server(
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
