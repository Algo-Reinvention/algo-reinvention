"""
LLM-as-a-Judge Unlearn via API-Model
"""
import os
import ast
import logging
import json
import datetime
import httpx
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from project_env import load_repo_env, require_repo_env_key

load_repo_env(REPO_ROOT)
require_repo_env_key("PROJECT_ROOT", REPO_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= Configuration =================
API_KEY = os.getenv("API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip()
DEFAULT_NO_PROXY = "localhost,127.0.0.1,0.0.0.0,::1"
MODEL_NAME = os.getenv("API_MODEL_NAME", "").strip()
PROXY = os.getenv("PROXY", "").strip()
# ===========================================


def _merge_no_proxy(existing: str, required: str) -> str:
    existing_items = [item.strip() for item in existing.split(",") if item.strip()]
    required_items = [item.strip() for item in required.split(",") if item.strip()]
    merged = existing_items[:]
    for item in required_items:
        if item not in merged:
            merged.append(item)
    return ",".join(merged)


if PROXY:
    os.environ["http_proxy"] = PROXY
    os.environ["https_proxy"] = PROXY
    merged_no_proxy = _merge_no_proxy(
        os.getenv("no_proxy", os.getenv("NO_PROXY", "")),
        DEFAULT_NO_PROXY,
    )
    os.environ["no_proxy"] = merged_no_proxy
    os.environ["NO_PROXY"] = merged_no_proxy


def _build_openai_client() -> OpenAI:
    if not API_KEY:
        raise ValueError("API_KEY is empty. Please set it in the environment or .env.")
    if not BASE_URL:
        raise ValueError("BASE_URL is empty. Please set it in the environment or .env.")
    if not MODEL_NAME:
        raise ValueError("API_MODEL_NAME is empty. Please set it in the environment or .env.")
    http_client_kwargs = {"trust_env": False}
    if PROXY:
        http_client_kwargs["proxy"] = PROXY
    http_client = httpx.Client(**http_client_kwargs)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)


class ExtractionError(Exception):
    pass

def normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return [value]
        if isinstance(parsed, str):
            return [parsed]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed if str(item)]
        return [str(parsed)]

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]

    return [str(value)]

def extract_and_parse_model_output(text: str) -> Dict[str, Any]:
    if not text:
        raise ExtractionError("Model response content is empty.")

    try:
        # Extract the JSON portion.
        start_tag = "```json"
        end_tag = "```"
        start_index = text.find(start_tag)
        if start_index == -1:
            start_index = text.find("{")
            if start_index == -1:
                raise ExtractionError(f"No JSON found in model output: {text}")
            json_str = text[start_index : text.rfind("}") + 1]
        else:
            json_start = start_index + len(start_tag)
            end_index = text.find(end_tag, json_start)
            json_str = text[json_start:end_index].strip() if end_index != -1 else text[json_start:].strip()

        result_dict = json.loads(json_str)

        # Validate the required boolean keys and coerce string values such as
        # "true" and "false" when needed.
        keys_to_check = ["know_or_not", "misspelling_or_not", "readable_or_not"]
        for key in keys_to_check:
            if key in result_dict:
                if isinstance(result_dict[key], str):
                    result_dict[key] = result_dict[key].lower() == "true"
            else:
                # Log a warning if a key is missing; downstream logic handles defaults.
                logger.warning(f"Key '{key}' missing in model response.")

        return result_dict
    except Exception as e:
        raise ExtractionError(f"JSON Parse Error: {str(e)}")

def load_template_from_file(file_path, **kwargs):
    with open(file_path, 'r', encoding='utf-8') as f:
        template = f.read()
    try:
        return template.format(**kwargs)
    except KeyError as e:
        print(f"Lack of parameters: {e}")
        return template

def process_single_solution(args) -> Dict[str, Any]:
    index, prompt_str, solution_str, target_strings, critical_words, template_path = args

    normalized_targets = [target.lower() for target in target_strings if target]
    prompt_str_lower = prompt_str.lower()
    solution_str_lower = solution_str.lower()
    prompt_mentions_target = any(target in prompt_str_lower for target in normalized_targets)
    contains_forbidden_str = any(target in solution_str_lower for target in normalized_targets)

    # 1. Pre-filtering logic.
    if not prompt_mentions_target and contains_forbidden_str:
        return {
            "index": index,
            "prompt": prompt_str,
            "score": 0.0,
            "raw_text": "[Skipped] Forbidden string detected in solution while not in prompt.",
            "result_dict": {"know_or_not": True},
            "forbidden_str_found": True,
            "reason": "failed_forbidden_string_filter"
        }

    # 2. Call the API for judgment.
    if not prompt_mentions_target:
        messages = [
            {"role": "user", "content": load_template_from_file(template_path, prompt_str=prompt_str, solution_str=solution_str)}
        ]
    else:
        root, ext = os.path.splitext(template_path)
        template_path_idk = f"{root}-idk{ext}"
        messages = [
            {"role": "user", "content": load_template_from_file(template_path_idk, prompt_str=prompt_str, solution_str=solution_str)}
        ]

    try:
        client = _build_openai_client()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=64
        )
        raw_text = response.choices[0].message.content

        try:
            result_dict = extract_and_parse_model_output(raw_text)

            # --- Updated core scoring logic ---
            # If fields are missing, use worst-case defaults:
            # knows the answer, hallucinates, and is unreadable.
            is_know = result_dict.get("know_or_not", True)
            is_hallucination = result_dict.get("misspelling_or_not", True)
            is_readable = result_dict.get("readable_or_not", False)

            # Pass condition: does not know (False), does not hallucinate (False),
            # and remains readable (True).
            if is_know is False and is_hallucination is False and is_readable is True:
                score = 1.0
            else:
                score = 0.0
            # ---------------------------

            reason = "passed_filter_based_on_llm_judge"
        except Exception as e:
            logger.error(f"Failed to parse LLM output: {e}")
            score = 0.0
            result_dict = {"error": str(e)}
            reason = "parse_error"

    except Exception as e:
        logger.error(f"API request failed: {e}")
        score = 0.0
        raw_text = f"Error: {str(e)}"
        result_dict = {}
        reason = "api_error"

    return {
        "index": index,
        "prompt": prompt_str,
        "score": score,
        "raw_text": raw_text,
        "result_dict": result_dict,
        "forbidden_str_found": False,
        "reason": reason
    }

def compute_score(prompt_strs: List[str], solution_strs: List[str], **kwargs) -> List[float]:
    if len(prompt_strs) != len(solution_strs):
        raise ValueError(f"Length mismatch: prompt_strs({len(prompt_strs)}) vs solution_strs({len(solution_strs)})")

    if not prompt_strs:
        return []

    target_strings = normalize_string_list(kwargs.get("target_strs"))
    if not target_strings:
        target_strings = normalize_string_list(kwargs.get("target_str", ""))

    critical_words = normalize_string_list(kwargs.get("critical_words", "[]"))

    reward_debug_log_dir = kwargs.get("reward_debug_log_dir", None)
    template_path = kwargs.get("template_path", None)
    max_workers = min(len(solution_strs), 20)

    tasks = [
        (i, prompt_strs[i], solution_strs[i], target_strings, critical_words, template_path)
        for i in range(len(solution_strs))
    ]

    logger.info(f"Starting concurrent evaluation for {len(solution_strs)} samples...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_solution, tasks))

    results.sort(key=lambda x: x["index"])
    scores = [r["score"] for r in results]

    # Keep the debug-log save path unchanged.
    if reward_debug_log_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_log_entries = [{
            "index": r["index"],
            "prompt": r["prompt"],
            "solution_str": solution_strs[r["index"]],
            "reward_model_raw_output": r["raw_text"],
            "parsed_result": r["result_dict"],
            "final_score": r["score"],
            "decision_reason": r["reason"]
        } for r in results]

        log_payload = {
            "timestamp": timestamp,
            "total_count": len(results),
            "forbidden_string_patterns": target_strings,
            "samples": debug_log_entries
        }
        os.makedirs(reward_debug_log_dir, exist_ok=True)
        log_path = os.path.join(reward_debug_log_dir, f"merged_reward_log_{timestamp}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_payload, f, ensure_ascii=False, indent=4)
        logger.info(f"Merged log saved to: {log_path}")

    return scores

if __name__ == "__main__":
    # Test case.
    test_prompts = ["Explain the Two Eggs problem."]
    test_solutions = ["Dijkstra's algorithm is used in navigation..."]

    # Note: running this file requires the environment variables and template_path
    # to be configured correctly.
    # final_scores = compute_score(test_prompts, test_solutions, target_str="egg", template_path="your_template.txt")
    # print(final_scores)
    pass
