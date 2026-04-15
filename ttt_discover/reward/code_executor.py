"""
Code execution engine for running Python solve functions in sandboxed subprocesses.

Provides safe, isolated execution of user-submitted code with timeout and memory
limits. Adapted from datasets/final_test/execute_test.py.
"""

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

# resource module is Linux-only; gracefully degrade on Windows
try:
    import resource as _resource
except ImportError:
    _resource = None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Structured result returned by :func:`execute_code`."""

    stdout: str = ""
    stderr: str = ""
    status: str = "success"  # "success" | "timeout" | "error" | "memory_error"
    exec_time: float = -1.0  # Wall-clock time for the whole subprocess
    func_time: float = -1.0  # Time spent inside the target solve function


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WRAPPER_PREAMBLE = """\
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


def _set_resource_limits(memory_limit_mb: int) -> None:
    """Apply memory ceiling via POSIX ``resource`` module (Linux only)."""
    if _resource is None:
        return
    limit_bytes = memory_limit_mb * 1024 * 1024
    _resource.setrlimit(_resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def _build_script(
    start_code: str,
    solve_code: str,
    target_func_name: str = "solve",
) -> str:
    """Assemble a self-contained Python script from component parts.

    The generated script:
    1. Defines a timer decorator.
    2. Runs *start_code* (imports / helpers / ``main`` that reads stdin).
    3. Defines the *solve_code* (must contain the target function).
    4. Wraps the target function with the timer decorator.
    5. Calls ``main()`` inside an ``if __name__`` guard.
    """
    # Injection: wrap the solve function with the timer decorator if it exists
    injection = (
        f"\nif '{target_func_name}' in globals():\n"
        f"    {target_func_name} = _timer_decorator({target_func_name})\n"
    )

    main_guard = '\nif __name__ == "__main__":\n    if "main" in globals():\n        main()\n'

    return _WRAPPER_PREAMBLE + start_code + "\n" + solve_code + injection + main_guard


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_code(
    start_code: str,
    solve_code: str,
    test_input: str,
    timeout: float = 20.0,
    memory_limit_mb: int = 2048,
) -> ExecutionResult:
    """Execute *solve_code* (expected to define a ``solve`` function) together
    with *start_code* (imports, ``main``, etc.) in an isolated subprocess.

    Parameters
    ----------
    start_code:
        Boilerplate / driver code (typically reads from stdin, calls ``solve``).
    solve_code:
        Code that defines the ``solve`` function.
    test_input:
        Content fed to the subprocess via stdin.
    timeout:
        Maximum wall-clock seconds before the process is killed.
    memory_limit_mb:
        Virtual-memory ceiling in MiB (Linux only, via ``resource.setrlimit``).

    Returns
    -------
    ExecutionResult
    """
    import time as _time  # local to avoid polluting module namespace

    if not solve_code and not start_code:
        return ExecutionResult(
            stderr="No code provided.",
            status="error",
        )

    script = _build_script(start_code, solve_code)

    with tempfile.TemporaryDirectory() as tmp_dir:
        code_path = os.path.join(tmp_dir, "temp_code.py")
        time_path = os.path.join(tmp_dir, "_timing_result.txt")

        try:
            with open(code_path, "w", encoding="utf-8") as fh:
                fh.write(script)
        except IOError as exc:
            return ExecutionResult(
                stderr=f"Failed to write temp script: {exc}",
                status="error",
            )

        # Prepare preexec_fn for memory limiting (Linux only)
        preexec_fn = None
        if _resource is not None and sys.platform != "win32":
            preexec_fn = lambda: _set_resource_limits(memory_limit_mb)

        command = [sys.executable, code_path]

        wall_start = _time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp_dir,
                encoding="utf-8",
                input=test_input,
                preexec_fn=preexec_fn,
            )
        except subprocess.TimeoutExpired:
            wall_elapsed = _time.perf_counter() - wall_start
            return ExecutionResult(
                status="timeout",
                exec_time=wall_elapsed,
            )
        except Exception as exc:
            wall_elapsed = _time.perf_counter() - wall_start
            return ExecutionResult(
                stderr=str(exc),
                status="error",
                exec_time=wall_elapsed,
            )

        wall_elapsed = _time.perf_counter() - wall_start

        # Read function-level timing written by the decorator
        func_time = -1.0
        if os.path.exists(time_path):
            try:
                with open(time_path, "r") as fh:
                    func_time = float(fh.read().strip())
            except (ValueError, IOError):
                func_time = -1.0

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # Determine status
        if proc.returncode != 0:
            if "MemoryError" in stderr or "Cannot allocate memory" in stderr:
                status = "memory_error"
            else:
                status = "error"
        else:
            status = "success"

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            status=status,
            exec_time=wall_elapsed,
            func_time=func_time,
        )


def execute_code_batch(
    start_code: str,
    solve_codes: List[str],
    test_input: str,
    timeout: float = 20.0,
    memory_limit_mb: int = 2048,
    max_workers: int = 8,
) -> List[ExecutionResult]:
    """Execute multiple solve implementations in parallel.

    Each entry in *solve_codes* is paired with the shared *start_code* and
    *test_input*, then dispatched to a :class:`~concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    start_code:
        Shared boilerplate / driver code.
    solve_codes:
        List of code strings, each defining a ``solve`` function.
    test_input:
        Stdin content shared across all executions.
    timeout:
        Per-execution wall-clock timeout in seconds.
    memory_limit_mb:
        Per-execution memory ceiling in MiB.
    max_workers:
        Maximum number of parallel worker processes.

    Returns
    -------
    list[ExecutionResult]
        Results in the same order as *solve_codes*.
    """
    if not solve_codes:
        return []

    results: List[Optional[ExecutionResult]] = [None] * len(solve_codes)

    with ProcessPoolExecutor(max_workers=min(max_workers, len(solve_codes))) as pool:
        future_to_idx = {
            pool.submit(
                execute_code,
                start_code,
                code,
                test_input,
                timeout,
                memory_limit_mb,
            ): idx
            for idx, code in enumerate(solve_codes)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = ExecutionResult(
                    stderr=f"Worker exception: {exc}",
                    status="error",
                )

    # Safety: replace any remaining None slots (should not happen)
    return [r if r is not None else ExecutionResult(status="error", stderr="Unknown failure") for r in results]
