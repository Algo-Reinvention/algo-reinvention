# tools/submit_final_answer.py
import os
import sys
import tempfile
import subprocess
import time
from typing import Dict, Any, Tuple


# --- Tool Definition (JSON Schema) ---

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "submit_final_answer",
        "description": "Call this function to submit the final Python code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The final Python code string will be used to solve the problem and compared to the gold standard.",
                }
            },
            "required": ["code"]
        }
    }
}

def submit_execute(
    code_parts: list,
    input_content: str,
    time_limit: float = 5,
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

    final_code = wrapper_logic + combined_code + injection + '\nif __name__ == "__main__":\n    main()'

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

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=time_limit,
                cwd=tmp_dir,
                encoding="utf-8",
                start_new_session=True,
                input=input_content
            )

            stdout = result.stdout
            stderr = result.stderr

            if result.returncode != 0:
                message = f"Execution failed. Return code: {result.returncode}.\nStderr: {stderr}"
            else:
                message = "Execution successful."
                if os.path.exists(time_file_path):
                    try:
                        with open(time_file_path, "r") as f:
                            target_func_time = float(f.read().strip())
                    except:
                        target_func_time = -1.0

        except subprocess.TimeoutExpired:
            message = f"Execution failed: Timeout after {time_limit} seconds."
        except Exception as e:
            message = f"Execution failed error: {e}"

        return stdout, stderr, message, target_func_time


# --- Tool Execution Function ---
def submit_final_answer_impl(fn_args: Dict[str, Any], task_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool entry point used by the model to submit the final answer code.
    Reads the code payload from fn_args.
    """
    # 1. Read the code submitted by the model from fn_args.
    code = fn_args.get('code', '')

    if not code.strip():
         return {"status": "error", "message": "Submission failed: Code is empty."}

    return {
        "status": "submission_received",
        "submitted_code": code # Keep the code in the result for multi_turn.py.
    }
