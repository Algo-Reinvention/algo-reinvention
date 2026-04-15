"""Load algorithm problems from the datasets/final_test directory.

Each problem directory (e.g. ``graph-sp-dijkstra``) contains:
* ``_generator/start_code.py`` – the boilerplate that wraps the submitted
  ``solve`` function.
* ``level0/``, ``level1/``, … – directories with numbered JSON problem files.

Every JSON file contains the full problem description and a list of
``test_cases`` with relative paths to input / ground-truth output files.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """A single algorithm problem instance."""

    problem_id: str
    """Unique identifier, e.g. ``graph-sp-dijkstra/level0/0``."""

    problem_text: str
    """Full problem description (Markdown) as stored in the JSON."""

    start_code: str
    """Boilerplate / submission template read from ``_generator/start_code.py``."""

    test_inputs: List[str] = field(default_factory=list)
    """Contents of every test-case input file referenced in the JSON."""

    test_outputs: List[str] = field(default_factory=list)
    """Contents of every corresponding ground-truth output file."""

    level: str = ""
    """Difficulty level label, e.g. ``level0``."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    """Read a text file, stripping a trailing newline if present."""
    return path.read_text(encoding="utf-8")


def _load_start_code(generator_dir: Path) -> str:
    """Return the content of ``start_code.py`` inside *generator_dir*.

    Returns an empty string (with a warning) when the file is missing.
    """
    start_code_path = generator_dir / "start_code.py"
    if not start_code_path.is_file():
        logger.warning("start_code.py not found at %s", start_code_path)
        return ""
    logger.debug("Reading start code from %s", start_code_path)
    return _read_text(start_code_path)


def _load_single_problem(
    project_root: Path,
    problem_dir_name: str,
    level: str,
    json_path: Path,
    start_code: str,
) -> Problem | None:
    """Parse one JSON problem file and resolve its test-case file paths.

    Returns ``None`` when the JSON cannot be parsed or is structurally
    unexpected so that the caller can skip it gracefully.
    """
    problem_id = f"{problem_dir_name}/{level}/{json_path.stem}"

    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read %s: %s", json_path, exc)
        return None

    problem_text: str = data.get("problem", "")
    if not problem_text:
        logger.warning("Empty problem text in %s", json_path)

    raw_test_cases: list[dict] = data.get("test_cases", [])

    test_inputs: list[str] = []
    test_outputs: list[str] = []

    for idx, tc in enumerate(raw_test_cases):
        input_rel = tc.get("input_path", "")
        output_rel = tc.get("output_path", "")

        if not input_rel or not output_rel:
            logger.warning(
                "Test case #%d in %s is missing input_path or output_path – skipping",
                idx,
                json_path,
            )
            continue

        input_path = project_root / input_rel
        output_path = project_root / output_rel

        if not input_path.is_file():
            logger.warning(
                "Test input file not found: %s (referenced by %s)",
                input_path,
                json_path,
            )
            continue

        if not output_path.is_file():
            logger.warning(
                "Test output file not found: %s (referenced by %s)",
                output_path,
                json_path,
            )
            continue

        try:
            test_inputs.append(_read_text(input_path))
            test_outputs.append(_read_text(output_path))
        except OSError as exc:
            logger.error(
                "Error reading test-case files for %s, case #%d: %s",
                json_path,
                idx,
                exc,
            )

    return Problem(
        problem_id=problem_id,
        problem_text=problem_text,
        start_code=start_code,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        level=level,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_problems(
    project_root: str,
    problem_dir: str,
    levels: list[str] | None = None,
) -> list[Problem]:
    """Load algorithm problems from the dataset.

    Parameters
    ----------
    project_root:
        Absolute (or relative) path to the top-level ``algo_test/`` directory
        that contains ``datasets/final_test/``.
    problem_dir:
        Name of the specific problem directory inside ``datasets/final_test/``
        (e.g. ``"graph-sp-dijkstra"``).  Pass ``"*"`` or ``""`` to load
        **all** problem directories found under ``datasets/final_test/``.
    levels:
        Which level sub-directories to scan, e.g.
        ``["level0", "level1", "level2"]``.  When *None* or empty, every
        ``levelN`` directory discovered on disk is used.

    Returns
    -------
    list[Problem]
        A (possibly empty) list of successfully loaded problems, sorted by
        ``problem_id``.
    """
    root = Path(project_root)
    dataset_root = root / "datasets/final_test"

    if not dataset_root.is_dir():
        logger.error("Dataset root does not exist: %s", dataset_root)
        return []

    # Resolve which problem directories to scan.
    if problem_dir and problem_dir != "*":
        problem_dirs = [dataset_root / problem_dir]
        if not problem_dirs[0].is_dir():
            logger.error("Problem directory not found: %s", problem_dirs[0])
            return []
    else:
        problem_dirs = sorted(
            p
            for p in dataset_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    problems: list[Problem] = []

    for pdir in problem_dirs:
        pdir_name = pdir.name

        # Skip non-problem entries (e.g. stray files).
        generator_dir = pdir / "_generator"
        if not generator_dir.is_dir():
            logger.debug(
                "Skipping %s – no _generator directory found", pdir_name
            )
            continue

        start_code = _load_start_code(generator_dir)

        # Determine which levels to iterate.
        if levels:
            level_dirs = [pdir / lvl for lvl in levels]
        else:
            # Auto-discover levelN directories.
            level_dirs = sorted(
                d for d in pdir.iterdir()
                if d.is_dir() and d.name.startswith("level")
            )

        if not level_dirs:
            logger.info(
                "No level directories found for problem %s – skipping",
                pdir_name,
            )
            continue

        for level_dir in level_dirs:
            if not level_dir.is_dir():
                logger.warning(
                    "Level directory does not exist: %s – skipping",
                    level_dir,
                )
                continue

            level_name = level_dir.name

            json_files = sorted(level_dir.glob("*.json"))
            if not json_files:
                logger.info(
                    "No JSON files in %s/%s – skipping",
                    pdir_name,
                    level_name,
                )
                continue

            for jf in json_files:
                problem = _load_single_problem(
                    project_root=root,
                    problem_dir_name=pdir_name,
                    level=level_name,
                    json_path=jf,
                    start_code=start_code,
                )
                if problem is not None:
                    problems.append(problem)

    logger.info(
        "Loaded %d problem(s) from %s (problem_dir=%r, levels=%r)",
        len(problems),
        dataset_root,
        problem_dir,
        levels,
    )
    return sorted(problems, key=lambda p: p.problem_id)
