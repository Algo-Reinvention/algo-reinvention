"""

cd ${PROJECT_ROOT}/datasets/final_test
python execute_test.py \
    --start-code-path string-manacher/_generator/start_code.py \
    --function-path  string-manacher/_generator/naive_sol_func.py \
    --input-path string-manacher/_generator/test_cases/alternating.txt

"""


# Save this file as, e.g., 'eval_runner.py'

import json
import subprocess
import tempfile
import os
import sys
import argparse
import time
import resource
from typing import Dict, Any, Tuple

def set_resource_limits(memory_limit_mb: int):
    if resource is None:
        return

    limit_in_bytes = memory_limit_mb * 1024 * 1024

    resource.setrlimit(resource.RLIMIT_AS, (limit_in_bytes, limit_in_bytes))

def submit_execute(
    code_parts: list,
    input_content: str,
    time_limit: float = 5,
    memory_limit_mb: int = 2048,
    target_func_name: str = "solve"
) -> Tuple[str, str, str, float]:

    if not code_parts:
        return "", "", "Code block not found or was empty.", -1.0

    wrapper_logic = f"""
import time
import functools

def _timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_t = time.perf_counter()
        res = func(*args, **kwargs)
        end_t = time.perf_counter()
        with open("_timing_result.txt", "w") as f:
            f.write(str(end_t - start_t))
        return res
    return wrapper
"""

    combined_code = "\n".join(code_parts)

    injection = f"\nif '{target_func_name}' in globals(): {target_func_name} = _timer_decorator({target_func_name})\n"

    final_code = wrapper_logic + combined_code + injection + '\nif __name__ == "__main__":\n    if "main" in globals(): main()'

    with tempfile.TemporaryDirectory() as tmp_dir:
        code_file_path = os.path.join(tmp_dir, "temp_code.py")
        time_file_path = os.path.join(tmp_dir, "_timing_result.txt")

        try:
            with open(code_file_path, "w", encoding="utf-8") as f:
                f.write(final_code)
        except IOError as e:
            return "", "", f"Failed to write code: {e}", -1.0

        command = [sys.executable, code_file_path]
        stdout, stderr, message = "", "", ""
        target_func_time = -1.0

        preexec_fn = None
        if resource is not None and sys.platform != "win32":
            preexec_fn = lambda: set_resource_limits(memory_limit_mb)
        else:
            pass

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=time_limit,
                cwd=tmp_dir,
                encoding="utf-8",
                input=input_content,
                preexec_fn=preexec_fn
            )

            stdout = result.stdout
            stderr = result.stderr

            if result.returncode != 0:
                msg = "Execution failed."
                if "MemoryError" in stderr:
                    msg = "Execution failed: Memory Limit Exceeded (MemoryError)."
                message = f"{msg} Return code: {result.returncode}.\nStderr: {stderr}"
            else:
                message = "Execution successful."
                if os.path.exists(time_file_path):
                    try:
                        with open(time_file_path, "r") as f:
                            target_func_time = float(f.read().strip())
                    except:
                        target_func_time = -1.0

        except subprocess.TimeoutExpired:
            message = f"Execution failed: Time Limit Exceeded (Timeout after {time_limit} seconds)."
        except Exception as e:
            message = f"Execution failed error: {e}"

        return stdout, stderr, message, target_func_time

def main():
    """
    Main function to parse arguments, read files, execute code, and print results.
    """
    parser = argparse.ArgumentParser(
        description="Execute a Python script with specified input and time limit for evaluation."
    )

    # Required arguments
    parser.add_argument(
        "--start-code-path",
        type=str,
        required=True,
        help="Path to the Python code file to be executed."
    )
    parser.add_argument(
        "--function-path",
        type=str,
        required=True,
        help="Path to the Python code file to be executed."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the TXT file containing the input content for stdin."
    )

    # Optional argument with default value 5.0
    parser.add_argument(
        "--time-limit",
        type=float,
        default=20.0,
        help="Execution time limit in seconds (default: 5.0)."
    )

    args = parser.parse_args()

    # 1. Read the code file
    try:
        with open(args.start_code_path, 'r', encoding='utf-8') as f:
            code_content_0 = f.read()
        with open(args.function_path, 'r', encoding='utf-8') as f:
            code_content_1 = f.read()
    except FileNotFoundError:
        print(json.dumps({"error": f"Error: Code file not found"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Error reading code file: {e}"}))
        sys.exit(1)

    # 2. Read the input file
    try:
        # Note: Input content is read as a string, which is correct for subprocess.run's input parameter.
        with open(args.input_path, 'r', encoding='utf-8') as f:
            input_content = f.read()
    except FileNotFoundError:
        print(json.dumps({"error": f"Error: Input file not found at {args.input_path}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Error reading input file {args.input_path}: {e}"}))
        sys.exit(1)

    # 3. Execute the code and measure time
    start_time = time.perf_counter()
    stdout, stderr, message, target_func_time = submit_execute(
        code_parts=[code_content_0, code_content_1],
        input_content=input_content,
        time_limit=args.time_limit
    )
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # 4. Prepare and print the structured output
    result_data = {
        "function_path": args.function_path,
        "input_path": args.input_path,
        "time_limit_set": args.time_limit,
        "execution_time_seconds": execution_time,
        "target_func_time": target_func_time,
        "stdout": stdout,
        "stderr": stderr,
        "message": message,
        "status": "Timeout" if "Timeout" in message else ("Error" if "failed" in message else "Success")
    }

    print(f"execution_time_seconds: {result_data['execution_time_seconds']}")

    # Print the result as a pretty-printed JSON object
    print(json.dumps(result_data, indent=4))

if __name__ == "__main__":
    main()
