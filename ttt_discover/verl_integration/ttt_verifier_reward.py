"""TTT-Discover verifier reward function for verl's BatchRewardManager.

This module provides a ``compute_score()`` function with the same signature as
the existing ``unlearn_reward_batch_api.py`` so that it can be plugged in via::

    custom_reward_function.path=.../ttt_verifier_reward.py
    custom_reward_function.name=compute_score

The function:
1. Extracts the ``solve`` function from each LLM response.
2. Runs it against the problem's test cases via the sandboxed code executor.
3. Returns the pass rate (0.0 – 1.0) as the scalar reward.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded problem cache (loaded once per process)
# ---------------------------------------------------------------------------

_PROBLEM_CACHE: Dict[str, object] = {}  # problem_id -> Problem
_CACHE_INITIALISED = False


def _ensure_problems_loaded(project_root: str, problem_dir: str, levels: Optional[str] = None) -> None:
    """Populate ``_PROBLEM_CACHE`` on first call.  Thread-safe-ish: worst case
    we load twice, but the dict-update is atomic in CPython.
    """
    global _CACHE_INITIALISED  # noqa: PLW0603

    if _CACHE_INITIALISED:
        return

    # Make sure the ttt_discover package is importable
    # (the project root should contain `ttt_discover/`)
    algo_test_root = os.path.dirname(project_root) if not os.path.isdir(
        os.path.join(project_root, "ttt_discover")
    ) else project_root

    if algo_test_root not in sys.path:
        sys.path.insert(0, algo_test_root)

    from ttt_discover.data.problem_loader import load_problems

    # Parse levels from comma-separated string or None
    level_list: Optional[list[str]] = None
    if levels:
        level_list = [l.strip() for l in levels.split(",") if l.strip()]

    problems = load_problems(
        project_root=project_root,
        problem_dir=problem_dir,
        levels=level_list,
    )

    for p in problems:
        _PROBLEM_CACHE[p.problem_id] = p

    _CACHE_INITIALISED = True
    logger.info(
        "TTT verifier reward: loaded %d problems from %s/%s (levels=%s)",
        len(problems),
        project_root,
        problem_dir,
        levels,
    )


# ---------------------------------------------------------------------------
# Public API — matches verl's compute_score signature
# ---------------------------------------------------------------------------

def compute_score(
    prompt_strs: List[str],
    solution_strs: List[str],
    **kwargs,
) -> List[float]:
    """Compute verifier rewards for a batch of LLM-generated solutions.

    Parameters
    ----------
    prompt_strs : list[str]
        Decoded prompt strings (unused here but required by verl's interface).
    solution_strs : list[str]
        Decoded response strings from the LLM.  Each should contain a
        ``def solve(...)`` function in a code block.
    **kwargs
        Must include:

        - ``project_root`` (str): path to the ``algo_test/`` directory.
        - ``problem_dir`` (str): e.g. ``"graph-sp-dijkstra"``.

        Optional:

        - ``levels`` (str): comma-separated level names, e.g. ``"level0,level1"``.
        - ``execution_timeout`` (float): per-test-case timeout in seconds (default 20).
        - ``extra_infos`` (list[dict]): each dict should have ``"problem_id"`` key.

    Returns
    -------
    list[float]
        One reward per sample (0.0 – 1.0).
    """
    # --- required kwargs ---
    project_root = kwargs.get("project_root", "")
    problem_dir = kwargs.get("problem_dir", "")
    levels = kwargs.get("levels", None)
    execution_timeout = float(kwargs.get("execution_timeout", 20.0))
    extra_infos: list = kwargs.get("extra_infos", [None] * len(solution_strs))

    if not project_root or not problem_dir:
        logger.error(
            "ttt_verifier_reward.compute_score: project_root=%r, problem_dir=%r — "
            "returning all zeros.",
            project_root,
            problem_dir,
        )
        return [0.0] * len(solution_strs)

    # --- lazy-load problems ---
    _ensure_problems_loaded(project_root, problem_dir, levels)

    # --- make ttt_discover importable (needed for extract_solve_function) ---
    algo_test_root = os.path.dirname(project_root) if not os.path.isdir(
        os.path.join(project_root, "ttt_discover")
    ) else project_root
    if algo_test_root not in sys.path:
        sys.path.insert(0, algo_test_root)

    from ttt_discover.utils.code_extraction import extract_solve_function
    from ttt_discover.reward.verifier import compute_verifier_reward

    # --- compute per-sample rewards ---
    scores: List[float] = []

    for i, solution_str in enumerate(solution_strs):
        # Determine which problem this sample corresponds to
        extra = extra_infos[i] if extra_infos[i] is not None else {}
        if isinstance(extra, dict):
            problem_id = extra.get("problem_id", "")
        else:
            problem_id = ""

        if not problem_id:
            logger.warning(
                "Sample %d: no problem_id in extra_info, returning 0.0", i
            )
            scores.append(0.0)
            continue

        problem = _PROBLEM_CACHE.get(problem_id)
        if problem is None:
            logger.warning(
                "Sample %d: problem_id=%r not found in cache (%d problems loaded), "
                "returning 0.0",
                i,
                problem_id,
                len(_PROBLEM_CACHE),
            )
            scores.append(0.0)
            continue

        # Extract the solve function from the LLM's response
        solve_code = extract_solve_function(solution_str)
        if not solve_code:
            logger.debug("Sample %d (problem %s): no solve function found", i, problem_id)
            scores.append(0.0)
            continue

        # Run against test cases
        try:
            reward = compute_verifier_reward(
                solve_code=solve_code,
                problem=problem,
                timeout=execution_timeout,
            )
            scores.append(reward)
        except Exception as exc:
            logger.error(
                "Sample %d (problem %s): verifier error: %s",
                i,
                problem_id,
                exc,
                exc_info=True,
            )
            scores.append(0.0)

    logger.info(
        "TTT verifier reward batch: %d samples, mean=%.4f",
        len(scores),
        sum(scores) / max(len(scores), 1),
    )
    return scores
