"""
Verifier reward module for TTT-Discover.

Executes candidate solve-code against a problem's test cases and computes
reward as the fraction of test cases whose stdout matches the expected output.

Typical usage::

    from algo_test.ttt_discover.reward.verifier import compute_verifier_reward
    from algo_test.ttt_discover.data.problem_loader import load_problems

    problems = load_problems(project_root, "graph-sp-dijkstra")
    reward = compute_verifier_reward(my_solve_code, problems[0])
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

from .code_executor import ExecutionResult, execute_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RewardResult:
    """Detailed result from :func:`compute_verifier_reward_detailed`."""

    reward: float = 0.0
    """Fraction of test cases that passed (0.0 – 1.0)."""

    test_case_results: List[dict] = field(default_factory=list)
    """Per-test-case detail dicts with keys:

    * ``input_idx`` – zero-based test-case index
    * ``passed``    – bool
    * ``expected_output`` – stripped expected string
    * ``actual_output``   – stripped actual stdout
    * ``status``    – execution status string
    * ``exec_time`` – wall-clock execution time (seconds)
    """

    execution_error: bool = False
    """True when *every* test case resulted in a non-success status."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_single_test(
    start_code: str,
    solve_code: str,
    test_input: str,
    expected_output: str,
    test_idx: int,
    timeout: float,
    memory_limit_mb: int,
) -> dict:
    """Execute *solve_code* on one test case and return a detail dict."""

    result: ExecutionResult = execute_code(
        start_code=start_code,
        solve_code=solve_code,
        test_input=test_input,
        timeout=timeout,
        memory_limit_mb=memory_limit_mb,
    )

    actual = result.stdout.strip()
    expected = expected_output.strip()
    passed = result.status == "success" and actual == expected

    return {
        "input_idx": test_idx,
        "passed": passed,
        "expected_output": expected,
        "actual_output": actual,
        "status": result.status,
        "exec_time": result.exec_time,
        "func_time": result.func_time,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_verifier_reward(
    solve_code: str,
    problem,  # Problem (imported type from data.problem_loader)
    timeout: float = 20.0,
    memory_limit_mb: int = 2048,
) -> float:
    """Compute reward as the pass rate across all test cases.

    Parameters
    ----------
    solve_code:
        Python source defining a ``solve`` function.
    problem:
        A :class:`~algo_test.ttt_discover.data.problem_loader.Problem` instance.
    timeout:
        Per-test-case wall-clock timeout in seconds.
    memory_limit_mb:
        Per-test-case memory ceiling in MiB.

    Returns
    -------
    float
        Fraction of test cases passed (0.0 – 1.0).
    """
    if not solve_code or not solve_code.strip():
        logger.warning(
            "Empty solve_code for problem %s – returning 0.0",
            problem.problem_id,
        )
        return 0.0

    total = len(problem.test_inputs)
    if total == 0:
        logger.warning(
            "Problem %s has no test cases – returning 0.0",
            problem.problem_id,
        )
        return 0.0

    passed_count = 0

    for idx in range(total):
        detail = _run_single_test(
            start_code=problem.start_code,
            solve_code=solve_code,
            test_input=problem.test_inputs[idx],
            expected_output=problem.test_outputs[idx],
            test_idx=idx,
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
        )

        if detail["passed"]:
            passed_count += 1
            logger.debug(
                "Problem %s, test %d/%d: PASSED (%.3fs)",
                problem.problem_id,
                idx + 1,
                total,
                detail["exec_time"],
            )
        else:
            logger.debug(
                "Problem %s, test %d/%d: FAILED (status=%s, "
                "expected=%r, got=%r)",
                problem.problem_id,
                idx + 1,
                total,
                detail["status"],
                detail["expected_output"][:80],
                detail["actual_output"][:80],
            )

    reward = passed_count / total
    logger.info(
        "Problem %s: reward=%.4f (%d/%d passed)",
        problem.problem_id,
        reward,
        passed_count,
        total,
    )
    return reward


def compute_verifier_reward_detailed(
    solve_code: str,
    problem,  # Problem
    timeout: float = 20.0,
    memory_limit_mb: int = 2048,
) -> RewardResult:
    """Like :func:`compute_verifier_reward` but returns full diagnostics.

    Parameters
    ----------
    solve_code:
        Python source defining a ``solve`` function.
    problem:
        A :class:`~algo_test.ttt_discover.data.problem_loader.Problem` instance.
    timeout:
        Per-test-case wall-clock timeout in seconds.
    memory_limit_mb:
        Per-test-case memory ceiling in MiB.

    Returns
    -------
    RewardResult
        Contains the scalar reward **and** per-test-case detail dicts.
    """
    if not solve_code or not solve_code.strip():
        logger.warning(
            "Empty solve_code for problem %s – returning empty RewardResult",
            problem.problem_id,
        )
        return RewardResult(reward=0.0, test_case_results=[], execution_error=True)

    total = len(problem.test_inputs)
    if total == 0:
        logger.warning(
            "Problem %s has no test cases – returning empty RewardResult",
            problem.problem_id,
        )
        return RewardResult(reward=0.0, test_case_results=[], execution_error=False)

    test_case_results: List[dict] = []
    passed_count = 0
    error_count = 0

    for idx in range(total):
        detail = _run_single_test(
            start_code=problem.start_code,
            solve_code=solve_code,
            test_input=problem.test_inputs[idx],
            expected_output=problem.test_outputs[idx],
            test_idx=idx,
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
        )
        test_case_results.append(detail)

        if detail["passed"]:
            passed_count += 1
            logger.debug(
                "Problem %s, test %d/%d: PASSED (%.3fs)",
                problem.problem_id,
                idx + 1,
                total,
                detail["exec_time"],
            )
        else:
            if detail["status"] != "success":
                error_count += 1
            logger.debug(
                "Problem %s, test %d/%d: FAILED (status=%s, "
                "expected=%r, got=%r)",
                problem.problem_id,
                idx + 1,
                total,
                detail["status"],
                detail["expected_output"][:80],
                detail["actual_output"][:80],
            )

    reward = passed_count / total
    execution_error = error_count == total

    logger.info(
        "Problem %s: reward=%.4f (%d/%d passed, %d errors, execution_error=%s)",
        problem.problem_id,
        reward,
        passed_count,
        total,
        error_count,
        execution_error,
    )

    return RewardResult(
        reward=reward,
        test_case_results=test_case_results,
        execution_error=execution_error,
    )


def compute_batch_rewards(
    solve_codes: List[str],
    problem,  # Problem
    timeout: float = 20.0,
    memory_limit_mb: int = 2048,
    max_workers: int = 8,
    reward_mode: str = "binary",
    time_limit: float = 0.9,
    time_ceiling: float = 1.5,
) -> List[float]:
    """Compute verifier rewards for multiple candidate solutions in parallel.

    Uses a :class:`~concurrent.futures.ThreadPoolExecutor` because each call
    to :func:`compute_verifier_reward` already spawns subprocesses via
    :func:`~.code_executor.execute_code`, so threads are sufficient for
    concurrency without adding process-spawn overhead.

    Parameters
    ----------
    solve_codes:
        List of *N* Python source strings, each defining a ``solve`` function.
    problem:
        A :class:`~algo_test.ttt_discover.data.problem_loader.Problem` instance
        shared across all evaluations.
    timeout:
        Per-test-case wall-clock timeout in seconds.
    memory_limit_mb:
        Per-test-case memory ceiling in MiB.
    max_workers:
        Maximum number of threads executing in parallel.
    reward_mode:
        ``"binary"`` – original pass-rate reward (0.0 or fraction).
        ``"continuous"`` – time-aware continuous reward: only correct answers
        receive non-zero scores, scaled by execution speed relative to
        *time_limit*.
    time_limit:
        Target time limit in seconds (only used when ``reward_mode="continuous"``).
        Correct answers faster than this get full reward (1.0).
    time_ceiling:
        Time ceiling in seconds (only used when ``reward_mode="continuous"``).
        Correct answers slower than this get zero reward (0.0).
        Between *time_limit* and *time_ceiling*, reward decays linearly.

    Returns
    -------
    list[float]
        *N* reward floats (0.0 – 1.0), in the same order as *solve_codes*.
    """
    n = len(solve_codes)
    if n == 0:
        return []

    logger.info(
        "Computing batch rewards for %d candidates on problem %s "
        "(max_workers=%d, mode=%s)",
        n,
        problem.problem_id,
        max_workers,
        reward_mode,
    )

    if reward_mode == "continuous":
        def _continuous_fn(code):
            return compute_continuous_reward(
                code, problem, timeout=timeout, memory_limit_mb=memory_limit_mb,
                time_limit=time_limit, time_ceiling=time_ceiling,
            )
        reward_fn = _continuous_fn
        is_continuous = True
    else:
        reward_fn = lambda code: compute_verifier_reward(
            code, problem, timeout=timeout, memory_limit_mb=memory_limit_mb,
        )
        is_continuous = False

    results: List = [None] * n

    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
        future_to_idx = {
            pool.submit(reward_fn, code): idx
            for idx, code in enumerate(solve_codes)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error(
                    "Batch reward worker %d raised: %s", idx, exc, exc_info=True
                )
                results[idx] = (0.0, -1.0) if is_continuous else 0.0

    # Unpack continuous results: (reward, max_func_time)
    if is_continuous:
        rewards = []
        max_func_times = []
        for r in results:
            if r is None:
                rewards.append(0.0)
                max_func_times.append(-1.0)
            elif isinstance(r, tuple):
                rewards.append(r[0])
                max_func_times.append(r[1])
            else:
                rewards.append(float(r))
                max_func_times.append(-1.0)

        # Check if ALL correct samples finish within 1.15s
        all_correct_within_threshold = True
        any_correct = False
        for reward_val, mft in zip(rewards, max_func_times):
            if reward_val > 0:  # this sample passed all tests
                any_correct = True
                if mft > 1.15 or mft < 0:
                    all_correct_within_threshold = False

        if any_correct and all_correct_within_threshold:
            logger.info(
                "★ Problem %s: ALL correct samples finish within 1.15s "
                "(max_func_times of correct samples: %s). "
                "The model can solve this problem correctly without needing reward=1.0.",
                problem.problem_id,
                [round(mft, 4) for mft, r in zip(max_func_times, rewards) if r > 0],
            )

        return rewards
    else:
        return [r if r is not None else 0.0 for r in results]


# ---------------------------------------------------------------------------
# Continuous reward (time-aware)
# ---------------------------------------------------------------------------

def _score_single_test_continuous(detail: dict, time_limit: float, time_ceiling: float) -> float:
    """Score a single test case with the continuous reward scheme.

    Scoring rules:
        - Not passed (error / timeout / wrong answer)          → 0.0
        - Passed and func_time <= time_limit                    → 1.0
        - Passed and time_limit < func_time < time_ceiling      → linear 1.0 → 0.0
        - Passed and func_time >= time_ceiling                  → 0.0
    """
    if not detail["passed"]:
        return 0.0

    func_time = detail.get("func_time", -1.0)

    # If timing is unavailable, fall back to exec_time
    if func_time < 0:
        func_time = detail.get("exec_time", -1.0)
    if func_time < 0:
        # Cannot determine time; give a middle-ground score for a correct answer
        return 0.5

    if func_time <= time_limit:
        return 1.0

    if func_time >= time_ceiling:
        return 0.0

    # Linear decay from 1.0 to 0.0 between time_limit and time_ceiling
    return 1.0 - (func_time - time_limit) / (time_ceiling - time_limit)


def compute_continuous_reward(
    solve_code: str,
    problem,  # Problem
    timeout: float = 20.0,
    memory_limit_mb: int = 2048,
    time_limit: float = 1.0,
    time_ceiling: float = 1.5,  # unused but kept for API compatibility
) -> float:
    """Compute a continuous reward based on correctness and execution speed.

    Reward design:
    * Code empty / extraction failed / runtime error / wrong answer → 0.0
    * All test cases correct, max_func_time ≤ 1s → 1.0
    * All test cases correct, max_func_time > 1s → 1.0 / max_func_time

    This gives a naturally continuous reward that is always > 0 for correct
    solutions, following the paper's convention of ``1/runtime`` style rewards.

    Parameters
    ----------
    solve_code:
        Python source defining a ``solve`` function.
    problem:
        Problem instance with test inputs / outputs.
    timeout:
        Wall-clock timeout for each test case execution.
    memory_limit_mb:
        Memory ceiling per execution.
    time_limit:
        Solutions faster than this get full reward (1.0). Default 1.0s.
    time_ceiling:
        Unused (kept for API compatibility with config).

    Returns
    -------
    tuple[float, float]
        (reward, max_func_time).  reward ∈ [0, 1].
    """
    if not solve_code or not solve_code.strip():
        return 0.0, -1.0

    total = len(problem.test_inputs)
    if total == 0:
        return 0.0, -1.0

    func_times: List[float] = []
    all_passed = True

    for idx in range(total):
        detail = _run_single_test(
            start_code=problem.start_code,
            solve_code=solve_code,
            test_input=problem.test_inputs[idx],
            expected_output=problem.test_outputs[idx],
            test_idx=idx,
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
        )

        if not detail["passed"]:
            all_passed = False
            # No need to run remaining tests — reward will be 0
            # But still record func_time for logging
            ft = detail.get("func_time", -1.0)
            if ft < 0:
                ft = detail.get("exec_time", -1.0)
            func_times.append(ft)
            break

        ft = detail.get("func_time", -1.0)
        if ft < 0:
            ft = detail.get("exec_time", -1.0)
        func_times.append(ft)

        logger.debug(
            "Problem %s, test %d/%d: passed=%s, status=%s, func_time=%.4f",
            problem.problem_id,
            idx + 1,
            total,
            detail["passed"],
            detail["status"],
            ft,
        )

    max_func_time = max(func_times) if func_times else -1.0

    if not all_passed:
        reward = 0.0
    elif max_func_time <= 0:
        raise RuntimeError(
            f"Problem {problem.problem_id}: all tests passed but "
            f"max_func_time={max_func_time} <= 0 (func_times={func_times}). "
            f"Timing data is missing or invalid."
        )
    else:
        # 1/t (no cap): e.g. 0.1s → 10.0, 0.5s → 2.0, 1s → 1.0, 2s → 0.5
        reward = time_limit / max_func_time

    logger.info(
        "Problem %s: continuous_reward=%.4f (all_passed=%s, max_func_time=%.4f)",
        problem.problem_id,
        reward,
        all_passed,
        max_func_time,
    )
    return reward, max_func_time
