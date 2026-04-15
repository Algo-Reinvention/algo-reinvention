import argparse
import concurrent.futures
import json
import os
import re
import time
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path
import sys
from urllib import error as urllib_error
from urllib import request as urllib_request

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from project_env import load_repo_env, require_repo_env_key

load_repo_env(REPO_ROOT)
require_repo_env_key("PROJECT_ROOT", REPO_ROOT)

from forget_test_utils import (
    prepare_forget_test_entry,
)


MODEL_FAMILY_QWEN3_INSTRUCT = "qwen3-instruct"
MODEL_FAMILY_MINISTRAL = "ministral"
MODEL_FAMILY_OTHER = "other"

DATASET_ALGO_NAME_MAP = {
    "array-gray": "Gray code",
    "array-moore": "Moore Voting",
    "graph-mst-prim": "Prim",
    "graph-sp-bellmanford": "Bellman-Ford",
    "graph-sp-dijkstra": "Dijkstra",
    "graph-sp-floyd": "Floyd",
    "math-euclidean": "Euclidean",
    "string-manacher": "Manacher",
    "string-kmp": "KMP",
    "math-strassen": "Strassen",
}

MINISTRAL_FORCE_SUFFIX = "[/THINK]Let's make the final answer:"
MINISTRAL_FORCE_MAX_TOKENS = 2048
DEFAULT_INFER_MODE = "batch"
DEFAULT_PRINT_PROMPT_IDS = False
DEFAULT_SHOW_PROGRESS = False
API_TIMEOUT_SEC = 120
API_MAX_RETRIES = 3
API_MAX_CONCURRENCY = 4
API_JUDGE_TEMPERATURE = 1.0
API_JUDGE_NUM_SAMPLES = 5
API_KEY = os.getenv("API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip()
MODEL_NAME = os.getenv("API_MODEL_NAME", "").strip()
PROXY = os.getenv("PROXY", "").strip()
JUDGE_PARSE_OK = "ok"
JUDGE_PARSE_MISSING_CODE_BLOCK = "missing_json_code_block"
JUDGE_PARSE_INVALID_JSON = "invalid_json_in_code_block"
JUDGE_PARSE_MISSING_KEYS = "missing_required_keys"
JUDGE_PARSE_NON_BOOLEAN = "non_boolean_values"
QUESTION_TYPE_MULTIPLE_CHOICE = "multiple_choice"
QUESTION_TYPE_DIRECT_ANSWER = "direct_answer"


def detect_model_family(model_path: str) -> str:
    p = (model_path or "").lower()
    if "ministral" in p:
        return MODEL_FAMILY_MINISTRAL
    if "qwen3" in p and "instruct" in p:
        return MODEL_FAMILY_QWEN3_INSTRUCT
    return MODEL_FAMILY_OTHER


def infer_algorithm_name_from_data_path(data_path: str) -> str:
    p = (data_path or "").lower()
    for key, algo_name in DATASET_ALGO_NAME_MAP.items():
        if key in p:
            return algo_name
    return "algorithm"


def build_qwen3_instruct_prefix(data_path: str) -> str:
    return "Let's solve the question carefully: "


def print_args(
    args: argparse.Namespace,
    program_name: str = None,
    version: str = None,
    show_version: bool = True,
) -> None:
    """print the args settings"""
    args_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}

    max_len = max(len(str(k)) for k in args_dict.keys())
    sep = "-" * (max_len + 20)

    output = []
    if program_name:
        output.append(f"\n\033[1;36m{program_name}\033[0m")

    if version and show_version:
        output.append(f"\033[1;34mVersion:\033[0m \033[1;33m{version}\033[0m")

    output.append(f"\033[1;35m{sep}\033[0m")

    for k, v in sorted(args_dict.items()):
        key = f"\033[1;32m{k:>{max_len}}\033[0m"
        val = f"\033[1;37m{str(v)}\033[0m"
        output.append(f"{key} : {val}")

    output.append(f"\033[1;35m{sep}\033[0m\n")

    print("\n".join(output))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Unlearning LLM with vLLM generation + API_MODEL judgement")
    parser.add_argument("--model_path", type=str, default="", help="Path to the trained model")
    parser.add_argument("--data_path", type=str, default="", help="Path to the input JSON data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSON results")
    parser.add_argument("--assistant_prefix", type=str, default="Let's analyze step by step: ", help="Optional prefix appended to generation prompt")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty to reduce runaway outputs")

    parser.add_argument(
        "--shuffle_options",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministically shuffle option order and remap labels before evaluation",
    )
    parser.add_argument("--shuffle_seed", type=int, default=20260211, help="Seed for deterministic option shuffling")
    parser.add_argument(
        "--skip_vllm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip vLLM generation and only run API_MODEL judgement on existing output_path JSON",
    )

    return parser.parse_args()


def build_prompt(tokenizer, question, question_type, assistant_prefix, data_path):
    rules = []
    if question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
        rules.append("Select exactly one option.")
        rules.append("Please place your final selection within the \\boxed{}.")
        instruction = "This is a multiple-choice test."
    else:
        # rules.append("Answer directly and briefly.")
        instruction = "This is a short-answer test."

    user_content = instruction + "\n" + "\n".join([f"- {r}" for r in rules]) + f"\n\n{question}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]

    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_prompt = full_prompt + assistant_prefix
    print(full_prompt)
    return full_prompt


def infer_algorithm_name_for_judge(question: str, data_path: str) -> str:
    by_path = infer_algorithm_name_from_data_path(data_path)
    if by_path != "algorithm":
        return by_path

    q = (question or "").lower()
    for candidate in DATASET_ALGO_NAME_MAP.values():
        if candidate.lower() in q:
            return candidate
    return "algorithm"


def scores_from_judge_flags(forget: bool):
    is_correct = not forget
    return {
        "correct": 1.0 if is_correct else 0.0,
        "forget": 1.0 if forget else 0.0,
    }


def extract_json_code_block(text: str) -> str:
    if not text:
        return ""
    matches = list(re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def _parse_bool_like(value):
    if isinstance(value, bool):
        return value, True
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value), True
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True, True
        if v == "false":
            return False, True
    return False, False


def parse_judge_response_content(content: str):
    result = {
        "forget": False,
        "parse_status": JUDGE_PARSE_OK,
        "parse_error": False,
        "parse_error_detail": "",
    }

    if not content:
        result["parse_status"] = JUDGE_PARSE_MISSING_CODE_BLOCK
        result["parse_error"] = True
        result["parse_error_detail"] = "empty_content"
        return result

    code_json = extract_json_code_block(content)
    if not code_json:
        result["parse_status"] = JUDGE_PARSE_MISSING_CODE_BLOCK
        result["parse_error"] = True
        result["parse_error_detail"] = "json_code_block_not_found"
        return result

    try:
        obj = json.loads(code_json)
    except Exception:
        result["parse_status"] = JUDGE_PARSE_INVALID_JSON
        result["parse_error"] = True
        result["parse_error_detail"] = "json_decode_failed"
        return result

    if not isinstance(obj, dict):
        result["parse_status"] = JUDGE_PARSE_INVALID_JSON
        result["parse_error"] = True
        result["parse_error_detail"] = "json_root_not_object"
        return result

    required_keys = ("forget",)
    if not all(key in obj for key in required_keys):
        result["parse_status"] = JUDGE_PARSE_MISSING_KEYS
        result["parse_error"] = True
        result["parse_error_detail"] = "missing_forget"
        return result

    parsed_flags = {}
    for key in required_keys:
        parsed_value, ok = _parse_bool_like(obj.get(key))
        if not ok:
            result["parse_status"] = JUDGE_PARSE_NON_BOOLEAN
            result["parse_error"] = True
            result["parse_error_detail"] = f"non_boolean_{key}"
            return result
        parsed_flags[key] = parsed_value

    result.update(parsed_flags)
    result["parse_status"] = JUDGE_PARSE_OK
    result["parse_error"] = False
    result["parse_error_detail"] = ""
    return result


def build_api_judge_messages(question: str, model_output: str, algorithm_name: str):
    system_prompt = "You are a strict computer science evaluator."
    user_prompt = f"""
<user_query>
{question}
</user_query>

<assistant_response>
{model_output}
</assistant_response>

Above is a user's query and an assistant's response. Please act as a computer science expert to evaluate whether the assistant's have forgotten the "{algorithm_name} algorithm".

Your evaluation should determine one specific attribute:

1. **forget (boolean)**:
   - Set to `true` if the assistant seems have never studied {algorithm_name} or you have not enough information to determine.
   - Note: If the response clearly names or accurately describes the {algorithm_name} algorithm, this must be `false`.

**Output Format**:
Please output your final evaluation as a JSON dictionary within a code block. Do not include any other text.

Example:
```json
{{
  "forget": false
}}
```
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_api_model_judge(question: str, model_output: str, algorithm_name: str):
    if not API_KEY:
        raise ValueError("API_KEY is empty. Please set API_KEY environment variable.")
    if not BASE_URL:
        raise ValueError("BASE_URL is empty. Please set BASE_URL environment variable.")
    if not MODEL_NAME:
        raise ValueError("API_MODEL_NAME is empty. Please set API_MODEL_NAME environment variable.")

    endpoint = f"{BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "temperature": API_JUDGE_TEMPERATURE,
        "max_tokens": 128,
        "messages": build_api_judge_messages(question, model_output, algorithm_name),
    }
    request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    req = urllib_request.Request(endpoint, data=request_bytes, headers=headers, method="POST")

    opener = urllib_request.build_opener()
    if PROXY:
        opener = urllib_request.build_opener(
            urllib_request.ProxyHandler({"http": PROXY, "https": PROXY})
        )

    last_error = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            with opener.open(req, timeout=API_TIMEOUT_SEC) as response:
                raw = response.read().decode("utf-8")
            response_obj = json.loads(raw)
            choices = response_obj.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content", "")
            parsed = parse_judge_response_content(content)
            return {
                "forget": parsed["forget"],
                "parse_status": parsed["parse_status"],
                "parse_error": parsed["parse_error"],
                "parse_error_detail": parsed["parse_error_detail"],
                "api_raw_content": content,
                "api_model": MODEL_NAME,
                "judge_algorithm": algorithm_name,
            }
        except urllib_error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"API HTTP error {e.code}: {error_body}")
        except urllib_error.URLError as e:
            last_error = RuntimeError(f"API URL error: {e.reason}")
        except Exception as e:
            last_error = RuntimeError(f"API parse/request error: {e}")

        if attempt < API_MAX_RETRIES:
            time.sleep(1.0)
            continue
        raise last_error


def build_summary_from_stats(stats):
    return {
        "avg_correct": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
        "avg_forget": stats["forget"] / stats["total"] if stats["total"] > 0 else 0,
        "majority_forget_rate": stats["forget_majority"] / stats["total"] if stats["total"] > 0 else 0,
        "total": stats["total"],
    }


def apply_scores_to_stats(stats, scores):
    stats["correct"] += scores["correct"]
    stats["forget"] += scores["forget"]
    stats["forget_majority"] += scores["forget_majority"]
    stats["total"] += 1


def evaluate_single_output_with_api(question, model_output, data_path, preset_algorithm=""):
    algorithm_name = preset_algorithm or infer_algorithm_name_for_judge(
        question=question,
        data_path=data_path,
    )

    sample_judges = []
    sample_scores = []

    for _ in range(API_JUDGE_NUM_SAMPLES):
        judge = call_api_model_judge(
            question=question,
            model_output=model_output,
            algorithm_name=algorithm_name,
        )
        sample_judges.append(judge)
        sample_scores.append(
            scores_from_judge_flags(
                forget=judge["forget"],
            )
        )

    n = float(len(sample_scores))
    forget_votes = sum(1 for j in sample_judges if j["forget"])
    forget_majority = 1.0 if forget_votes > (n / 2.0) else 0.0
    scores = {
        "correct": sum(s["correct"] for s in sample_scores) / n,
        "forget": sum(s["forget"] for s in sample_scores) / n,
        "forget_majority": forget_majority,
    }

    parse_status_counter = Counter(j["parse_status"] for j in sample_judges)
    majority_parse_status = parse_status_counter.most_common(1)[0][0] if parse_status_counter else JUDGE_PARSE_OK
    parse_error_count = sum(1 for j in sample_judges if j["parse_error"])
    non_empty_details = [j["parse_error_detail"] for j in sample_judges if j["parse_error_detail"]]

    judge_fields = {
        "judge_forget": scores["forget"],
        "judge_forget_majority": bool(forget_majority),
        "judge_sample_count": int(n),
        "judge_parse_status": majority_parse_status,
        "judge_parse_error": parse_error_count > 0,
        "judge_parse_error_detail": "; ".join(non_empty_details[:3]),
        "judge_parse_error_count": parse_error_count,
        "judge_model": MODEL_NAME,
        "judge_algorithm": algorithm_name,
        "judge_raw_content": sample_judges[0]["api_raw_content"] if sample_judges else "",
        "judge_raw_contents": [j["api_raw_content"] for j in sample_judges],
        "judge_parse_statuses": [j["parse_status"] for j in sample_judges],
        "judge_decision_source": "api",
        "judge_shortcut_keyword": "",
        "judge_prompt_mentions_algorithm": False,
    }
    return judge_fields, scores


def evaluate_batch_with_api(jobs, show_progress=False):
    results = [None] * len(jobs)
    stats = {"correct": 0.0, "forget": 0.0, "forget_majority": 0.0, "total": 0}
    if not jobs:
        return [], stats

    max_workers = min(API_MAX_CONCURRENCY, len(jobs))

    def _run_one(job):
        judge_fields, scores = evaluate_single_output_with_api(
            question=job["question"],
            model_output=job["model_output"],
            data_path=job["data_path"],
            preset_algorithm=job.get("preset_algorithm", ""),
        )
        record = dict(job["base_record"])
        record.update(judge_fields)
        record["scores"] = scores
        return record, scores

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_run_one, job): idx
            for idx, job in enumerate(jobs)
        }
        completed = concurrent.futures.as_completed(future_to_idx)
        for future in tqdm(completed, total=len(future_to_idx), disable=not show_progress):
            idx = future_to_idx[future]
            record, scores = future.result()
            results[idx] = record
            apply_scores_to_stats(stats, scores)

    return results, stats


def evaluate_existing_details_with_api(details, default_data_path):
    jobs = []

    for detail in details:
        if not isinstance(detail, dict):
            continue
        question = detail.get("question") or detail.get("presented_question") or ""
        full_output = detail.get("full_output", "")

        if not full_output and isinstance(detail.get("sample_outputs"), list) and detail["sample_outputs"]:
            full_output = detail["sample_outputs"][0]

        merged = dict(detail)
        merged.pop("sample_outputs", None)
        merged.pop("sample_extracted_answers", None)
        merged.pop("num_samples", None)
        merged.pop("judge_unsolved", None)
        if isinstance(merged.get("scores"), dict):
            merged["scores"].pop("unresolved", None)
        merged["full_output"] = full_output
        jobs.append(
            {
                "question": question,
                "model_output": full_output,
                "data_path": default_data_path,
                "preset_algorithm": detail.get("judge_algorithm", ""),
                "base_record": merged,
            }
        )

    return evaluate_batch_with_api(jobs, show_progress=DEFAULT_SHOW_PROGRESS)


def run_text_eval_vllm(args, prepared_data, tokenizer):
    """Run vLLM generation once per question, then judge via API_MODEL."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("Please install vllm: pip install vllm")

    infer_mode = DEFAULT_INFER_MODE
    print_prompt_ids = DEFAULT_PRINT_PROMPT_IDS
    show_progress = DEFAULT_SHOW_PROGRESS

    model_family = detect_model_family(args.model_path)
    dynamic_prefix = ""
    # dynamic_prefix = build_qwen3_instruct_prefix(args.data_path)
    # if model_family == MODEL_FAMILY_QWEN3_INSTRUCT:
    #     print(f"Detected qwen3-instruct model. Injecting prefix before generation: {dynamic_prefix}")
    # elif model_family == MODEL_FAMILY_MINISTRAL:
    #     print(
    #         "Detected ministral model. If finish_reason is 'length', "
    #         "will force-close thinking and continue generation for final answer."
    #     )

    print(f"Initializing vLLM model from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=max(torch.cuda.device_count(), 1),
    )

    first_sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    second_sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=MINISTRAL_FORCE_MAX_TOKENS,
        repetition_penalty=args.repetition_penalty,
    )

    def generate_with_optional_progress(prompts, sampling_params):
        try:
            return llm.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=show_progress,
            )
        except TypeError as e:
            # Backward compatibility for vLLM versions without `use_tqdm`.
            if "use_tqdm" not in str(e):
                raise
            if show_progress:
                return llm.generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                )
            with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
                return llm.generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                )

    def force_continue_ministral_if_needed(prompt_ids, generated_text, generated_ids):
        suffix = ""
        suffix_ids = []
        if "[/THINK]" not in generated_text:
            suffix = MINISTRAL_FORCE_SUFFIX
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

        continued_prompt_ids = prompt_ids + generated_ids + suffix_ids
        second_outputs = generate_with_optional_progress(
            prompts=[{"prompt_token_ids": continued_prompt_ids}],
            sampling_params=second_sampling_params,
        )

        second_text = ""
        if second_outputs and second_outputs[0].outputs:
            second_text = second_outputs[0].outputs[0].text or ""
        return generated_text + suffix + second_text

    prompt_token_ids_batch = []
    effective_assistant_prefix = args.assistant_prefix + dynamic_prefix
    for i, entry in enumerate(prepared_data):
        prompt_text = build_prompt(
            tokenizer,
            entry["question"],
            entry["question_type"],
            effective_assistant_prefix,
            args.data_path,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if print_prompt_ids:
            print(f"[PromptIDs][Item {i}] len={len(prompt_ids)}")
            print(prompt_ids)
        prompt_token_ids_batch.append(prompt_ids)

    outputs = []
    if infer_mode == "batch":
        print(f"Generating responses for {len(prompt_token_ids_batch)} items with vLLM batch inference...")
        tokenized_prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids_batch]
        outputs = generate_with_optional_progress(
            prompts=tokenized_prompts,
            sampling_params=first_sampling_params,
        )
    else:
        print(f"Generating responses for {len(prompt_token_ids_batch)} items with vLLM single-item inference...")
        for prompt_ids in tqdm(prompt_token_ids_batch, disable=not show_progress):
            single_outputs = generate_with_optional_progress(
                prompts=[{"prompt_token_ids": prompt_ids}],
                sampling_params=first_sampling_params,
            )
            outputs.append(single_outputs[0] if single_outputs else None)

    jobs = []
    for q_idx, output in enumerate(tqdm(outputs, disable=not show_progress)):
        entry = prepared_data[q_idx]

        if output is not None and output.outputs:
            primary = output.outputs[0]
            generated_text = primary.text or ""
            finish_reason = primary.finish_reason
            if getattr(primary, "token_ids", None) is not None:
                generated_ids = list(primary.token_ids)
            else:
                generated_ids = tokenizer.encode(generated_text, add_special_tokens=False)
        else:
            generated_text = ""
            finish_reason = None
            generated_ids = []

        if model_family == MODEL_FAMILY_MINISTRAL and finish_reason == "length":
            print(f"[Ministral][Item {q_idx}] hit token limit, forcing final answer continuation...")
            generated_text = force_continue_ministral_if_needed(
                prompt_ids=prompt_token_ids_batch[q_idx],
                generated_text=generated_text,
                generated_ids=generated_ids,
            )

        base_record = {
            "question": entry["original_question"],
            "presented_question": entry["question"],
            "question_type": entry["question_type"],
            "label_mapping": entry["label_mapping"],
            "unknown_labels": entry["unknown_labels"],
            "full_output": generated_text,
        }
        jobs.append(
            {
                "question": entry["question"],
                "model_output": generated_text,
                "data_path": args.data_path,
                "preset_algorithm": "",
                "base_record": base_record,
            }
        )

    return evaluate_batch_with_api(jobs, show_progress=show_progress)


def main():
    args = parse_args()

    print_args(args)

    if args.skip_vllm:
        if not os.path.exists(args.output_path):
            raise FileNotFoundError(f"--skip_vllm is set but output_path does not exist: {args.output_path}")
        with open(args.output_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

        if isinstance(existing_data, dict):
            existing_details = existing_data.get("details", [])
            output_data = dict(existing_data)
        elif isinstance(existing_data, list):
            existing_details = existing_data
            output_data = {}
        else:
            raise ValueError("Existing output JSON must be a dict with `details` or a list of detail records.")

        default_data_path = args.data_path
        if not default_data_path and isinstance(existing_data, dict):
            cfg = existing_data.get("config", {})
            if isinstance(cfg, dict):
                default_data_path = cfg.get("data_path", "")

        final_results, final_stats = evaluate_existing_details_with_api(
            existing_details,
            default_data_path=default_data_path,
        )
        summary = build_summary_from_stats(final_stats)
        mode = "API_JUDGE_ONLY"
    else:
        if not args.model_path:
            raise ValueError("`--model_path` is required when --skip_vllm is False.")
        if not args.data_path:
            raise ValueError("`--data_path` is required when --skip_vllm is False.")

        print(f"Loading tokenizer from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        with open(args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prepared_data = [
            prepare_forget_test_entry(
                entry=entry,
                idx=i,
                shuffle_options=args.shuffle_options,
                shuffle_seed=args.shuffle_seed,
            )
            for i, entry in enumerate(data)
        ]

        final_results, final_stats = run_text_eval_vllm(args, prepared_data, tokenizer)
        summary = build_summary_from_stats(final_stats)
        output_data = {}
        mode = "VLLM_TEXT+API_JUDGE"

    output_data["config"] = vars(args)
    output_data["summary"] = summary
    output_data["details"] = final_results

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\nEvaluation Complete! Mode: {mode}")
    print(
        f"Summary: Correct: {summary['avg_correct']:.4f}, "
        f"Forget: {summary['avg_forget']:.4f}, "
        f"MajorityForgetRate: {summary['majority_forget_rate']:.4f}"
    )
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
