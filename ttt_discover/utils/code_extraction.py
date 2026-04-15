"""
Extract Python solve functions from LLM-generated text responses.

Handles various formats: fenced code blocks, raw code, mixed
explanation/code, and multiple candidate functions.
"""

import logging
import re

logger = logging.getLogger(__name__)


def extract_code_blocks(text: str) -> list[str]:
    """Extract all ```python ... ``` or ``` ... ``` code blocks from text.

    Matches fenced blocks that use either the ``python`` language tag or
    no language tag at all.  The content between the fences is returned
    with leading/trailing whitespace stripped.

    Args:
        text: Raw text potentially containing fenced code blocks.

    Returns:
        A list of code-block contents (may be empty).
    """
    # Match ```python ... ``` and ``` ... ``` (with optional language tag).
    # re.DOTALL so '.' matches newlines inside the block.
    pattern = r"```(?:python|py)?\s*\n(.*?)```"
    blocks = [match.strip() for match in re.findall(pattern, text, re.DOTALL)]

    if blocks:
        logger.debug("Found %d fenced code block(s)", len(blocks))
    else:
        logger.debug("No fenced code blocks found in text")

    return blocks


def extract_function_by_name(code: str, func_name: str = "solve") -> str:
    """Extract a function definition from *code* by name.

    Walks the source lines, finds ``def <func_name>(`` (with optional
    decorator lines immediately above), and then collects every subsequent
    line that belongs to the function body by tracking indentation.

    Handles:
    - Decorators stacked above the ``def`` line.
    - Nested ``def`` statements inside the function body.
    - Blank lines interspersed in the body.
    - Varying indentation depths.

    Args:
        code: A string of Python source code.
        func_name: The function name to search for (default ``"solve"``).

    Returns:
        The full function source (decorators + def + body) as a string,
        or an empty string if the function is not found.
    """
    lines = code.splitlines()
    # We collect *all* matches and return the last one (see docstring of
    # extract_solve_function for the rationale).
    matches: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Detect `def func_name(`
        if re.match(rf"def\s+{re.escape(func_name)}\s*\(", stripped):
            # ------ gather decorator lines immediately above ------
            decorator_start = i
            j = i - 1
            while j >= 0:
                prev_stripped = lines[j].lstrip()
                if prev_stripped.startswith("@"):
                    decorator_start = j
                    j -= 1
                elif prev_stripped == "":
                    # Allow blank lines between decorators and def only if
                    # a decorator sits above the blank line.
                    k = j - 1
                    while k >= 0 and lines[k].strip() == "":
                        k -= 1
                    if k >= 0 and lines[k].lstrip().startswith("@"):
                        decorator_start = j
                        j -= 1
                    else:
                        break
                else:
                    break

            # ------ determine the indentation of the def line ------
            def_indent = len(line) - len(line.lstrip())

            # ------ collect function body ------
            func_lines = list(lines[decorator_start : i + 1])  # decorators + def
            i += 1

            while i < len(lines):
                cur = lines[i]

                # Blank / whitespace-only lines are part of the body
                if cur.strip() == "":
                    func_lines.append(cur)
                    i += 1
                    continue

                cur_indent = len(cur) - len(cur.lstrip())
                if cur_indent > def_indent:
                    # Still inside the function body
                    func_lines.append(cur)
                    i += 1
                else:
                    # Reached a line at the same (or lesser) indent -> function ended.
                    break

            # Strip trailing blank lines that don't really belong to the func.
            while func_lines and func_lines[-1].strip() == "":
                func_lines.pop()

            matches.append("\n".join(func_lines))
            # Don't increment i here -- the outer loop will re-examine the
            # current line (it may be the start of another function).
            continue

        i += 1

    if not matches:
        logger.debug("Function '%s' not found in provided code", func_name)
        return ""

    if len(matches) > 1:
        logger.debug(
            "Found %d definitions of '%s'; using the last one",
            len(matches),
            func_name,
        )

    return matches[-1]


def extract_solve_function(response: str) -> str:
    """Extract the ``solve`` function code from model output.

    Strategy (in order of priority):

    1. **Fenced code blocks** -- look for ```python ... ``` (or bare
       ``` ... ```).  If any block contains ``def solve(``, use it.
       When multiple blocks qualify, take the *last* one (most likely to
       be the final/corrected version).
    2. **Raw code** -- if no fenced block is found, search the entire
       response text for ``def solve(`` directly.
    3. **Indentation tracking** -- once the ``def`` line is located,
       ``extract_function_by_name`` walks forward collecting every line
       whose indentation is deeper than the ``def`` line.

    Args:
        response: The full model response text.

    Returns:
        The extracted solve function code as a string.
        Returns an empty string if no solve function is found.
    """
    if not response or not response.strip():
        logger.warning("Empty response provided to extract_solve_function")
        return ""

    # ---- Strategy 1: fenced code blocks ----
    code_blocks = extract_code_blocks(response)

    if code_blocks:
        # Prefer blocks that actually contain `def solve(`
        candidates = [b for b in code_blocks if re.search(r"def\s+solve\s*\(", b)]

        if candidates:
            # Pick the last qualifying block (final/corrected version).
            chosen_block = candidates[-1]
            logger.debug(
                "Using fenced code block %d/%d containing 'def solve'",
                len(candidates),
                len(code_blocks),
            )
            result = extract_function_by_name(chosen_block, "solve")
            if result:
                return result
            # The block itself contains `def solve(` but our extractor
            # failed (shouldn't happen, but be safe).  Fall back to
            # returning the whole block.
            logger.warning(
                "Fenced block matched 'def solve' but extraction failed; "
                "returning full code block"
            )
            return chosen_block

        # No block contains `def solve` -- try extracting from each block
        # anyway (the user might have named it differently and we can
        # still give it our best shot).  But per the spec we only look
        # for `solve`, so fall through to Strategy 2.
        logger.debug(
            "Fenced code blocks found but none contain 'def solve'; "
            "falling back to raw-text search"
        )

    # ---- Strategy 2: raw code in the response ----
    if re.search(r"def\s+solve\s*\(", response):
        logger.debug("Found 'def solve(' in raw response text (no fenced block)")
        result = extract_function_by_name(response, "solve")
        if result:
            return result
        logger.warning(
            "'def solve(' found in response but function extraction failed"
        )

    # ---- Nothing found ----
    logger.warning("No solve function found in model response")
    return ""
