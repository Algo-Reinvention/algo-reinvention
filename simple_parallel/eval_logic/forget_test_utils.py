import random
import re
from typing import Dict, List, Sequence, Tuple

OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E")
QUESTION_TYPE_MULTIPLE_CHOICE = "multiple_choice"
QUESTION_TYPE_DIRECT_ANSWER = "direct_answer"

_UNKNOWN_OPTION_PATTERNS = (
    re.compile(r"\bi\s*don'?t\s*know\b", flags=re.IGNORECASE),
    re.compile(r"\bi\s*don'?t\s*remember\b", flags=re.IGNORECASE),
    re.compile(r"\bunknown\b", flags=re.IGNORECASE),
    re.compile(r"\bnot\s+above\b", flags=re.IGNORECASE),
    re.compile(r"\bnone\s+of\s+the\s+above\b", flags=re.IGNORECASE),
    re.compile(r"\bmight\s+not\s+exist\b", flags=re.IGNORECASE),
)

try:
    from math_verify import parse as mv_parse, verify as mv_verify
except Exception:  # pragma: no cover - optional dependency fallback
    mv_parse = None
    mv_verify = None


def _normalize_option_label(raw_text: str, candidate_labels: Sequence[str]) -> str:
    if not raw_text:
        return ""
    candidate_set = {label.upper() for label in candidate_labels}
    text = raw_text.strip()

    # Common boxed variants: \text{A}, \mathrm{A}, etc.
    wrapped = re.search(r"\\(?:text|mathrm|operatorname)\s*\{\s*([A-Za-z])\s*\}", text)
    if wrapped:
        label = wrapped.group(1).upper()
        if label in candidate_set:
            return label

    token_match = re.search(r"(?<![A-Za-z0-9_])([A-Za-z])(?![A-Za-z0-9_])", text)
    if token_match:
        label = token_match.group(1).upper()
        if label in candidate_set:
            return label

    return ""


def _extract_braced_content(text: str, open_brace_idx: int) -> str:
    depth = 0
    start = -1
    for i in range(open_brace_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth == 1:
                start = i + 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i]
    return ""


def extract_last_boxed_content(text: str) -> str:
    if not text:
        return ""
    starts = [m.end() - 1 for m in re.finditer(r"\\boxed\s*\{", text)]
    for open_brace_idx in reversed(starts):
        content = _extract_braced_content(text, open_brace_idx)
        if content:
            return content.strip()
    return ""


def _extract_choice_by_patterns(text: str, candidate_labels: Sequence[str]) -> str:
    if not text:
        return ""
    labels = "".join(sorted({label.upper() for label in candidate_labels}))
    if not labels:
        labels = "A-E"

    tail = text[-1600:]
    patterns = [
        rf"(?:final|correct)\s+answer\s*(?:is|:)?\s*[\(\[]?\s*([{labels}])\s*[\)\]]?",
        rf"answer\s*(?:is|:)\s*[\(\[]?\s*([{labels}])\s*[\)\]]?",
        rf"choose\s+(?:option\s*)?([{labels}])\b",
        rf"option\s*([{labels}])\b",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, tail, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()

    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    for line in reversed(lines):
        m = re.fullmatch(rf"[\(\[]?\s*([{labels}])\s*[\)\].:!? ]*", line, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return ""


def _extract_choice_by_math_verify(text: str, candidate_labels: Sequence[str]) -> str:
    if not text or mv_parse is None or mv_verify is None:
        return ""
    try:
        parsed = mv_parse(text)
    except Exception:
        return ""
    if not parsed:
        return ""

    for label in [c.upper() for c in candidate_labels]:
        try:
            target = mv_parse(f"\\boxed{{{label}}}")
            if target and mv_verify(parsed, target):
                return label
        except Exception:
            continue
    return ""


def extract_choice_label(text: str, candidate_labels: Sequence[str] = OPTION_LABELS) -> str:
    boxed_content = extract_last_boxed_content(text)
    label = _normalize_option_label(boxed_content, candidate_labels)
    if label:
        return label

    label = _extract_choice_by_patterns(text, candidate_labels)
    if label:
        return label

    return _extract_choice_by_math_verify(text, candidate_labels)


def _split_question_options(question: str) -> Tuple[str, List[Tuple[str, str]]]:
    labels = "".join(OPTION_LABELS)
    option_start = re.search(rf"(?:^|\n)\s*([{labels}])\.\s", question)
    if not option_start:
        return "", []

    start_idx = option_start.start(1)
    stem = question[:start_idx].rstrip()
    option_block = question[start_idx:]

    matches = list(re.finditer(rf"(?<![A-Za-z0-9_])([{labels}])\.\s", option_block))
    if len(matches) < 2:
        return "", []
    options: List[Tuple[str, str]] = []

    for i, match in enumerate(matches):
        letter = match.group(1).upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(option_block)
        option_text = option_block[start:end].strip()
        options.append((letter, option_text))

    return stem, options


def _format_question(stem: str, options: Sequence[Tuple[str, str]]) -> str:
    option_part = " ".join([f"{label}. {text}" for label, text in options])
    if stem:
        return f"{stem}\n{option_part}"
    return option_part


def detect_unknown_option_labels(options: Sequence[Tuple[str, str]]) -> List[str]:
    labels = []
    for label, text in options:
        if any(pattern.search(text or "") for pattern in _UNKNOWN_OPTION_PATTERNS):
            labels.append(label)
    return labels


def normalize_question_type(value: str, has_options: bool) -> str:
    raw = (value or "").strip().lower().replace("-", "_")
    if raw in {QUESTION_TYPE_MULTIPLE_CHOICE, QUESTION_TYPE_DIRECT_ANSWER}:
        return raw
    return QUESTION_TYPE_MULTIPLE_CHOICE if has_options else QUESTION_TYPE_DIRECT_ANSWER


def prepare_forget_test_entry(
    entry: Dict,
    idx: int,
    shuffle_options: bool,
    shuffle_seed: int,
) -> Dict:
    question = entry["question"]

    stem, options = _split_question_options(question)

    question_type = normalize_question_type(entry.get("question_type", ""), has_options=bool(options))

    if question_type != QUESTION_TYPE_MULTIPLE_CHOICE or not options:
        label_mapping = {label: label for label in OPTION_LABELS}
        unknown_labels: List[str] = []
        return {
            "original_question": question,
            "question": question,
            "question_type": question_type,
            "label_mapping": label_mapping,
            "unknown_labels": unknown_labels,
        }

    if shuffle_options:
        shuffled = options[:]
        random.Random(shuffle_seed + idx).shuffle(shuffled)
        relabeled_options: List[Tuple[str, str]] = []
        label_mapping: Dict[str, str] = {}
        for new_label, (old_label, text) in zip(OPTION_LABELS, shuffled):
            relabeled_options.append((new_label, text))
            label_mapping[old_label] = new_label
    else:
        relabeled_options = options[:]
        label_mapping = {old_label: old_label for old_label, _ in options}

    if shuffle_options:
        remapped_question = _format_question(stem, relabeled_options)
        unknown_labels = detect_unknown_option_labels(relabeled_options)
    else:
        remapped_question = question
        unknown_labels = detect_unknown_option_labels(relabeled_options)

    return {
        "original_question": question,
        "question": remapped_question,
        "question_type": question_type,
        "label_mapping": label_mapping,
        "unknown_labels": unknown_labels,
    }
