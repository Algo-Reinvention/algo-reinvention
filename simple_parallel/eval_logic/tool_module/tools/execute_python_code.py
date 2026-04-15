# tools/execute_python_code.py
import json
import subprocess
import tempfile
import os
import sys
from typing import Dict, Any, Tuple


# --- Tool Definition (JSON Schema) ---

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "execute_python_code",
        "description": "This tool is used to execute Python code snippets in a test environment for debugging and verifying intermediate logic. The execution result will be returned as feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The string of Python code to be executed (must be a complete, runnable Python script).",
                },
                "input_content": {
                    "type": "string",
                    "description": "The text content passed to the code as Stdin input during code execution.",
                }
            },
            "required": ["code", "input_content"]
        }
    }
}

def execute_python_code(code: str, input_content: str, time_limit: float) -> Tuple[str, str, str]:
    """
    Executes the given Python code in a temporary directory, passing input_content via stdin.
    Returns (stdout, stderr, execution_output_message).
    """
    if not code:
        return "", "", "Code block not found or was empty."

    # Use 'with tempfile.TemporaryDirectory()' to ensure cleanup
    with tempfile.TemporaryDirectory() as tmp_dir:
        code_file_path = os.path.join(tmp_dir, "temp_code.py")
        
        # Write the code to a file
        try:
            with open(code_file_path, "w", encoding="utf-8") as f:
                f.write(code)
        except IOError as e:
            return "", "", f"Failed to write code to temporary file: {e}"

        # Construct the command (only the python interpreter and the script path)
        command = [sys.executable, code_file_path]
        
        stdout = ""
        stderr = ""
        
        try:
            # Run the code as a subprocess
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=time_limit,
                cwd=tmp_dir, # Run in the temporary directory
                encoding="utf-8",
                start_new_session=True,
                # Pass the input_content to the subprocess's standard input (stdin)
                input=input_content 
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            if result.returncode != 0:
                # Execution finished with an error (non-zero return code)
                # Ensure the message clearly contains the stderr for feedback
                message = f"Execution failed with return code {result.returncode}.\nStderr: {stderr}"
            else:
                # Execution finished successfully
                message = "Execution successful."

        except subprocess.TimeoutExpired:
            # Process exceeded the time limit
            message = f"Execution failed: Timeout after {time_limit} seconds."
            stdout = ""
            stderr = ""
        except Exception as e:
            # Other exceptions during subprocess creation/execution
            message = f"Execution failed due to system environment error: {e}"
            stdout = ""
            stderr = ""

        return stdout, stderr, message


# --- Tool Execution Function ---
def execute_python_code_impl(fn_args: Dict[str, Any], task_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Python code for model-side debugging.
    Read model-provided arguments from fn_args and task configuration from task_config.
    """
    # 1. Read model-provided arguments from fn_args.
    code = fn_args.get('code', '')
    # Default to the input configured in the task.
    input_content = fn_args.get('input_content', task_config.get('default_input_content', '')) 
    
    # 2. Read task-level configuration values.
    time_limit = task_config.get('time_limit', 5.0) 
    
    if not code.strip():
        return {
            "status": "error",
            "message": "Execution failed: Code block was empty.",
            "stdout": "",
            "stderr": ""
        }

    # 3. Invoke the core execution helper directly.
    stdout, stderr, message = execute_python_code(code, input_content, time_limit)
    
    # 4. Derive the final status from the message returned by execute_python_code.
    # Note: this relies on matching specific message strings.
    status = "success"
    if "Timeout" in message:
        status = "timeout"
    elif "Execution failed with return code" in message:
        status = "runtime_error"
    elif "Failed to write code" in message or "internal error" in message:
        # Covers write failures or internal exceptions.
        status = "internal_error"

    return {
        "status": status,
        "message": message,
        "stdout": stdout,
        "stderr": stderr,
    }
