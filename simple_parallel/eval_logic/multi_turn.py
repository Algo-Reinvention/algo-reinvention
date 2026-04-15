"""multi-turn tool-use reasoning

Logs:
- 2025.12.09: current only support Qwen
- 2025.12.25: Updated to support multiple test cases (Scheme A: All-Pass)
"""
import sys
import os
import argparse
import json
import time
import re
import subprocess
import traceback
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from project_env import load_repo_env, require_repo_env_key

load_repo_env(repo_root)
require_repo_env_key("PROJECT_ROOT", repo_root)

from framework.register import register_processor
from utils.util import *
from .infer_module.infer_vllm_server import LLM_Service
from vllm import SamplingParams

from .tool_module.tools.submit_final_answer import submit_execute
from .tool_module.tool_registry import get_tool_definitions
from .tool_module.tool_utils import process_tool_calls, parse_raw_text_for_tool_calls


_cached_llm_service = None

API_TIMEOUT_SEC = 120
API_MAX_RETRIES = 3
API_VERIFY_TEMPERATURE = 0.2
API_VERIFY_MAX_TOKENS = 16384
API_PROXY_URL = os.getenv("PROXY", "").strip()
EVAL_DEBUG_LOG_DIR_ENV = "FINAL_TEST_DEBUG_LOG_DIR"
EVAL_DEBUG_TEXT_PREVIEW_CHARS = 512


def _resolve_api_model_name(base_url: str) -> str:
    if _is_local_base_url(base_url):
        return "model"
    model_name = os.getenv("API_MODEL_NAME", "").strip()
    if not model_name:
        raise ValueError("API_MODEL_NAME is empty. Please set API_MODEL_NAME environment variable.")
    return model_name


def _is_local_base_url(base_url: str) -> bool:
    try:
        parsed = urlparse((base_url or "").strip())
    except Exception:
        return False

    host = (parsed.hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _build_verify_messages(problem: str, submission_status: Dict[str, Any]) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a strict code submission verifier. "
        "Never output, hint at, or mention the correct solution. "
        "Only explain why the provided submission failed."
    )
    status_text = json.dumps(submission_status, ensure_ascii=False)
    user_prompt = f"""
<problem>
{problem}
</problem>

Given the problem background, please check why the following submission fail to pass. Pay attention that you must not output or mention the correct solution, you should only explain why the submission fail:
<submission_status>
{status_text}
</submission_status>
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_verify_content(raw_content: Any) -> str:
    if isinstance(raw_content, list):
        text_parts = []
        for item in raw_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif item is not None:
                text_parts.append(str(item))
        raw_content = "".join(text_parts)
    elif raw_content is None:
        raw_content = ""
    elif not isinstance(raw_content, str):
        raw_content = str(raw_content)

    content = raw_content.strip()
    if not content:
        return ""

    marker_found = False
    for marker in ("</think>",):
        marker_index = content.rfind(marker)
        if marker_index != -1:
            marker_found = True
            post_think_content = content[marker_index + len(marker):].strip()
            if post_think_content:
                return post_think_content
    if marker_found:
        return ""
    return content


def _should_append_post_submit_assistant_ack(
    model_path: str,
    is_submission: bool,
    next_user_content: str,
) -> bool:
    model_path_lower = (model_path or "").lower()
    return (
        is_submission
        and "ministral" in model_path_lower
        and "instruct" in model_path_lower
        and bool((next_user_content or "").strip())
    )


MINISTRAL_SUBMIT_TOOL_PREFIX = "[TOOL_CALLS]submit_final_answer[ARGS]"


def _build_post_no_submission_assistant_prefix(
    model_path: str,
    next_user_content: str,
) -> Optional[str]:
    model_path_lower = (model_path or "").lower()
    normalized_content = (next_user_content or "").strip()
    if (
        "ministral" in model_path_lower
        and "instruct" in model_path_lower
        and normalized_content == "No submission detected. Please use submit_final_answer."
    ):
        return MINISTRAL_SUBMIT_TOOL_PREFIX
    return None


def _merge_assistant_prefix_for_tool_parsing(
    raw_text: str,
    assistant_prefix: Optional[str],
) -> str:
    normalized_raw_text = "" if raw_text is None else str(raw_text)
    normalized_prefix = (assistant_prefix or "").strip()
    if normalized_prefix.startswith("[TOOL_CALLS]"):
        return normalized_prefix + normalized_raw_text
    return normalized_raw_text


def _summarize_debug_text(text: Any) -> Dict[str, Any]:
    normalized = "" if text is None else str(text)
    return {
        "chars": len(normalized),
        "preview": normalized[:EVAL_DEBUG_TEXT_PREVIEW_CHARS],
    }


def _maybe_save_eval_debug_record(record: Dict[str, Any], stream_id: int, turn: int) -> Optional[str]:
    debug_dir = (os.getenv(EVAL_DEBUG_LOG_DIR_ENV) or "").strip()
    if not debug_dir:
        return None

    os.makedirs(debug_dir, exist_ok=True)
    file_path = os.path.join(
        debug_dir,
        f"eval_debug_{int(time.time() * 1000)}_s{stream_id}_t{turn}.json",
    )
    save_json(record, file_path)
    return file_path


def call_api_model_verify(problem: str, submission_status: Dict[str, Any]) -> Dict[str, str]:
    base_url = os.getenv("BASE_URL", "").strip()
    if not base_url:
        raise ValueError("BASE_URL is empty. Please set BASE_URL environment variable.")

    api_key = os.getenv("API_KEY", "").strip()
    local_base_url = _is_local_base_url(base_url)
    print(base_url)
    print(api_key)
    print(local_base_url)
    if not api_key and not local_base_url:
        raise ValueError("API_KEY is empty for non-local verifier endpoint.")

    model_name = _resolve_api_model_name(base_url)
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "temperature": API_VERIFY_TEMPERATURE,
        "max_tokens": API_VERIFY_MAX_TOKENS,
        "messages": _build_verify_messages(problem=problem, submission_status=submission_status),
    }
    request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib_request.Request(endpoint, data=request_bytes, headers=headers, method="POST")

    opener = urllib_request.build_opener()
    if API_PROXY_URL and not local_base_url:
        opener = urllib_request.build_opener(
            urllib_request.ProxyHandler({"http": API_PROXY_URL, "https": API_PROXY_URL})
        )

    last_error = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            with opener.open(req, timeout=API_TIMEOUT_SEC) as response:
                raw = response.read().decode("utf-8")
            response_obj = json.loads(raw)
            choices = response_obj.get("choices") or []
            first_choice = choices[0] if choices else {}
            message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
            finish_reason = first_choice.get("finish_reason", "") if isinstance(first_choice, dict) else ""
            if finish_reason == "length":
                return {
                    "content": "",
                    "api_model": model_name,
                    "skipped": True,
                    "skip_reason": "max_tokens_exceeded",
                }

            content = _extract_verify_content(message.get("content", ""))
            print(payload)
            print(content)
            if not content.strip():
                raise RuntimeError("API_MODEL returned empty verify content.")
            return {"content": content, "api_model": model_name}
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

def get_llm_service(model_path: str = None, tensor_parallel_size: int = 1, server_url: str = "http://localhost:8000") -> LLM_Service:
    global _cached_llm_service
    if _cached_llm_service is None:
        timestamped_print("MODEL: First call to get_llm_service. Attempting to connect to vLLM server...")
        try:
            _cached_llm_service = LLM_Service(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                server_url=server_url,
                use_server=True
            )
            timestamped_print("MODEL: Connected to vLLM server successfully via get_llm_service.")
        except Exception as e:
            timestamped_print(f"MODEL: Failed to connect to vLLM server via get_llm_service: {e}", "ERROR")
            raise ValueError(f"Failed to connect to vLLM server: {e}")
    else:
        timestamped_print("MODEL: vLLM service already cached, reusing connection.")
    return _cached_llm_service

def compare_outputs(expected: str, actual: str) -> Tuple[bool, str]:
    """Compare expected and actual outputs strictly but ignoring trailing whitespace"""

    comparison = expected.strip() == actual.strip()
    msg = "Output matches ground truth." if comparison else "Output mismatch."
    return comparison, msg

@register_processor('parse_process_args')
def parse_eval_args(remaining_args: argparse.Namespace):
    parser = argparse.ArgumentParser(description="Response Generation Args.")
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--tensor_parallel_size", type=int, help="Number of tensor parallelism.")

    # Generation parameters
    parser.add_argument("--num", type=int, default=1, help="Number of responses to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=7000, help="Maximum tokens to generate")
    parser.add_argument("--logprobs", type=int, default=0, help="Number of most likely tokens to return logprobs for (0 to disable)")
    parser.add_argument("--disable_thinking",action="store_true", help="Disable internal thinking process or verbose logging.")

    # Code execution parameters
    parser.add_argument("--time_limit", type=float, default=5.0, help="Time limit in seconds for EACH test case execution.")
    parser.add_argument("--target_time_limit", type=float, default=1.0, help="Time limit in seconds for target function execution.")
    parser.add_argument("--tools", type=lambda s: s.split(','), default=None,
                        help="Comma-separated list of tool names to enable (e.g., 'execute_code,submit_final_answer'). Default is all tools.")

    # === final_test argument ===
    parser.add_argument("--final_test", action="store_true",
                        help="If set, executes the final submitted code against ALL test cases and evaluates correctness.")
    parser.add_argument("--gen_verify", action="store_true",
                        help="If set, call API_MODEL verifier for failed submissions with non-empty output.")
    parser.add_argument("--start_code_path", type=str,
                        default="")

    # Prompts
    parser.add_argument("--system_prompt", type=str,
                        default="You are a research scientist. Please call the execute_python_code tool to make sure that your code is correct, then call the submit_final_answer tool to submit the final code.",
                        help="System prompt for the model.")
    parser.add_argument("--user_prompt_template", type=str,
                        default="{question}",
                        help="User prompt template.")
    parser.add_argument("--assistant_prefix", type=str,
                        default="",
                        help="Assistant prefix.")

    # Server configuration
    parser.add_argument("--server_url", type=str, default="http://localhost:8000",
                        help="vLLM server URL (Base URL)")

    # Iterative Self-Correction Parameters
    parser.add_argument("--max_turn", type=int, default=30,
                        help="Maximum number of generation/execution turns for self-correction.")

    return parser.parse_args(remaining_args)

@register_processor('check_finish')
def check_finish(args: argparse.Namespace, save_path: str) -> bool:
    if not os.path.exists(save_path):
        return False
    try:
        record = load_json(save_path)
        if 'generated_responses' not in record:
            return False
        if len(record['generated_responses']) < args.num:
            timestamped_print(f"Not enough response streams in {save_path}. Expected {args.num}, got {len(record['generated_responses'])}.", "WARNING")
            return False
        return True
    except Exception as e:
        timestamped_print(f"Error loading JSON file {save_path}: {e}", "ERROR")
        return False


# @codex: Capture stable exception metadata so failed runs still persist actionable debugging context.
def _extract_error_info(exc: Exception) -> Dict[str, Any]:
    tb_entries = traceback.extract_tb(exc.__traceback__)
    location: Dict[str, Any] = {}
    if tb_entries:
        last = tb_entries[-1]
        location = {
            "file": last.filename,
            "line": last.lineno,
            "function": last.name,
            "code": last.line,
        }

    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "location": location,
        "traceback": traceback.format_exc(),
    }


def _build_result_data(
    args: argparse.Namespace,
    question: str,
    loaded_test_cases: List[Dict[str, Any]],
    all_response_streams: List[Dict[str, Any]],
    final_context: str,
    termination_reason: Dict[str, Any] = None,
    error_info: Dict[str, Any] = None,
) -> Dict[str, Any]:
    result_data = {
        'problem': question,
        'test_cases_count': len(loaded_test_cases),
        'timestamp': time.time(),
        'generation_config': {
            'num': args.num,
            'max_turn': args.max_turn,
            'final_test': args.final_test,
            'gen_verify': bool(getattr(args, 'gen_verify', False)),
            'temperature': args.temperature,
            'time_limit': args.time_limit
        },
        'generated_responses': all_response_streams,
        'final_context': final_context,
    }
    if termination_reason is not None:
        result_data['termination_reason'] = termination_reason
    if error_info is not None:
        result_data['error'] = error_info
    return result_data


@register_processor('process')
def process_file(args) -> None:
    args = fix_cli_newlines(args)
    question = ""
    loaded_test_cases: List[Dict[str, Any]] = []
    all_tools: List[Dict[str, Any]] = []
    all_response_streams: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = []
    llm_service = None
    termination_reason: Dict[str, Any] = None
    current_stream_id = 0
    current_turn = 0

    try:
        # Load question data
        data = load_json(args.input_filepath)

        question = data.get('question', '')

        if args.final_test:
            test_cases_config = data.get('test_cases', [])

            if not test_cases_config:
                raise ValueError("--final_test is enabled but no 'test_cases' found in input data.")

            timestamped_print(f"Pre-loading {len(test_cases_config)} test cases...")

            for idx, case in enumerate(test_cases_config):
                i_path_rel = case.get('input_path')
                o_path_rel = case.get('output_path')
                if not i_path_rel or not o_path_rel:
                    raise ValueError(f"Test case {idx} missing input_path or output_path.")

                i_path = os.path.join(args.project_root, i_path_rel)
                o_path = os.path.join(args.project_root, o_path_rel)

                with open(i_path, 'r', encoding='utf-8') as f:
                    i_content = f.read()
                with open(o_path, 'r', encoding='utf-8') as f:
                    o_content = f.read()

                loaded_test_cases.append({
                    "input": i_content,
                    "gt": o_content,
                    "id": idx + 1,
                    "input_path": i_path_rel,
                    "ground_truth_path": o_path_rel,
                })
            timestamped_print(f"Successfully loaded {len(loaded_test_cases)} test cases.")

        default_input_content = ""
        if loaded_test_cases:
            default_input_content = loaded_test_cases[0]['input']
        with open(os.path.join(args.project_root, args.start_code_path), 'r', encoding='utf-8') as f:
            start_code = f.read()

        # Get LLM service
        llm_service = get_llm_service(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            server_url=args.server_url
        )

        if not llm_service.tokenizer:
            raise ValueError("LLM_Service did not load a tokenizer. Cannot track tokens.")

        all_tools = get_tool_definitions(tool_names=args.tools)
        enabled_tool_names = [t['function']['name'] for t in all_tools]
        timestamped_print(f"Enabled Tools: {', '.join(enabled_tool_names)}")

        user_prompt = args.user_prompt_template.format(question=question)
        if args.system_prompt != "":
            initial_messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            initial_messages = [
                {"role": "user", "content": user_prompt}
            ]

        timestamped_print(f"Starting generation for {args.num} streams, with max_turn={args.max_turn}...")
        stop_all_streams = False
        gen_verify_enabled = bool(getattr(args, "gen_verify", False))

        # Loop for streams
        for stream_id in range(args.num):
            current_stream_id = stream_id + 1
            timestamped_print(f"--- Stream {stream_id + 1}/{args.num} started ---")

            messages = initial_messages[:]
            # @codex: 4 means we intentionally stop this file because assistant produced an empty turn.
            is_correct = 0 # 0: ongoing, 1: correct, -1: max_tokens, 2: submitted(no-exec), 3: submitted(fail), 4: empty assistant
            current_turn = 1
            total_completion_tokens = 0
            temp_assistant_prefix = None

            while current_turn <= args.max_turn and (is_correct == 0 or is_correct == 3):
                timestamped_print(f"Stream {stream_id + 1}: Turn {current_turn}/{args.max_turn} generating...")

                current_prompt = llm_service.build_prompt(messages=messages, tools=all_tools, disable_thinking=args.disable_thinking)
                if temp_assistant_prefix is None:
                    current_prompt += args.assistant_prefix
                else:
                    current_prompt += temp_assistant_prefix

                prompt_tokens = len(llm_service.tokenizer.encode(current_prompt, add_special_tokens=False))
                remaining_max_tokens = args.max_tokens - prompt_tokens

                if remaining_max_tokens <= 0:
                    is_correct = -1
                    break

                sampling_params = SamplingParams(
                    n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    max_tokens=remaining_max_tokens, logprobs=args.logprobs, stop=["<|im_end|>"]
                )

                try:
                    outputs = llm_service.inference(prompt=current_prompt, sampling_params=sampling_params)
                except Exception as e:
                    timestamped_print(f"Stream {stream_id + 1}: Inference failed: {e}", "ERROR")
                    break

                if not outputs:
                    # @codex: Empty assistant turn is terminal to avoid invalid Ministral chat-template state.
                    is_correct = 4
                    termination_reason = {
                        "type": "empty_assistant_response",
                        "message": "No model output returned for assistant turn.",
                        "stream_id": stream_id + 1,
                        "turn": current_turn,
                    }
                    timestamped_print(
                        f"Stream {stream_id + 1}: Empty assistant output detected at turn {current_turn}. Ending and saving.",
                        "WARNING",
                    )
                    stop_all_streams = True
                    break

                raw_text = outputs[0].get('text') or ""
                finish_reason = outputs[0].get('finish_reason', 'stop')
                active_assistant_prefix = temp_assistant_prefix
                raw_text_for_tool_parsing = _merge_assistant_prefix_for_tool_parsing(
                    raw_text=raw_text,
                    assistant_prefix=active_assistant_prefix,
                )
                generated_content, tool_calls = parse_raw_text_for_tool_calls(
                    raw_text_for_tool_parsing,
                    args.model_path,
                )
                generated_content = generated_content or ""

                if generated_content.strip() == "" and not tool_calls:
                    # @codex: Prevent appending invalid empty assistant messages that crash chat template rendering.
                    is_correct = 4
                    termination_reason = {
                        "type": "empty_assistant_response",
                        "message": "Assistant content is empty and no tool calls were returned.",
                        "stream_id": stream_id + 1,
                        "turn": current_turn,
                    }
                    timestamped_print(
                        f"Stream {stream_id + 1}: Empty assistant content detected at turn {current_turn}. Ending and saving.",
                        "WARNING",
                    )
                    stop_all_streams = True
                    break

                if active_assistant_prefix is None:
                    message_obj = {
                        "role": "assistant",
                        "content": args.assistant_prefix + generated_content
                    }
                else:
                    message_obj = {
                        "role": "assistant",
                        "content": active_assistant_prefix + generated_content
                    }
                    temp_assistant_prefix = None

                if tool_calls:
                    message_obj["tool_calls"] = tool_calls

                generated_tokens_in_turn = len(llm_service.tokenizer.encode(raw_text, add_special_tokens=False))
                total_completion_tokens += generated_tokens_in_turn

                if finish_reason == 'length':
                    is_correct = -1
                    timestamped_print(f"Stream {stream_id + 1}: Max tokens reached.", "WARNING")

                messages.append(message_obj)

                is_submission = False
                submission_data = {}
                end_flag = False
                tool_messages = []
                verify_user_content = ""

                if tool_calls:
                    task_config = {
                        'time_limit': args.time_limit,
                        'default_input_content': default_input_content,
                    }

                    tool_messages, tool_results = process_tool_calls(tool_calls=tool_calls, task_config=task_config)

                    if 'submit_final_answer' in tool_results:
                        is_submission = True
                        submission_data = tool_results['submit_final_answer']

                    if is_submission:
                        submitted_code = submission_data.get('submitted_code', '')

                        if not args.final_test:
                            timestamped_print(f"Stream {stream_id + 1}: Submission detected. --final_test is False. Stopping.", "INFO")
                            is_correct = 2
                            end_flag = True
                            comparison_msg = "Submission received (Evaluation skipped)."
                        else:
                            timestamped_print(f"Stream {stream_id + 1}: Executing FINAL SUBMISSION against {len(loaded_test_cases)} test cases...")

                            all_passed = True
                            failed_case_info = ""
                            max_time_used = 0.0
                            target_times = []
                            case_details_log = []
                            for case in loaded_test_cases:
                                case_stdout, case_stderr, case_msg, case_time = submit_execute(
                                    code_parts=[start_code, submitted_code],
                                    input_content=case['input'],
                                    time_limit=args.time_limit
                                )

                                case_details_log.append({
                                    "case_id": case['id'],
                                    "input_path": case.get('input_path', ''),
                                    "ground_truth_path": case.get('ground_truth_path', ''),
                                    "stdout": _summarize_debug_text(case_stdout),
                                    "stderr": _summarize_debug_text(case_stderr),
                                    "exec_msg": case_msg,
                                    "time_used": case_time
                                })

                                max_time_used = max(max_time_used, case_time)
                                target_times.append(case_time)
                                if case_time < 0:
                                    max_time_used = 100  # set to a large time

                                if "Execution successful" not in case_msg:
                                    all_passed = False
                                    failed_case_info = f"Runtime Error on Case {case['id']}: {case_msg.splitlines()[0]}"
                                    break

                                match, _ = compare_outputs(case['gt'], case_stdout)
                                if not match:
                                    all_passed = False
                                    if case_stdout.strip() != "":
                                        failed_case_info = f"Wrong Answer on Case {case['id']}"
                                    else:
                                        failed_case_info = f"No output detected. Make sure that you print the answer explicitly in function solve()."
                                    break

                            if all_passed:
                                if max_time_used < args.target_time_limit:
                                    is_correct = 1
                                    comparison_msg = f"Accepted! Passed all {len(loaded_test_cases)} cases. Max Time: {max_time_used:.4f}s. Target Times: {target_times}"
                                    timestamped_print(f"Stream {stream_id + 1}: {comparison_msg}", "SUCCESS")
                                    end_flag = True
                                else:
                                    is_correct = 3 # submitted but incorrect
                                    comparison_msg = f"Failed: Timeout! Please check your time complexity carefully"
                                    timestamped_print(f"Stream {stream_id + 1}: {comparison_msg}", "FAILURE")
                            else:
                                is_correct = 3 # submitted but incorrect
                                comparison_msg = f"Failed: {failed_case_info}"
                                timestamped_print(f"Stream {stream_id + 1}: {comparison_msg}", "FAILURE")

                            temp_eval_record = {
                                "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "stream_id": stream_id + 1,
                                "turn": current_turn,
                                "is_correct": is_correct,
                                "comparison_msg": comparison_msg,
                                "start_code": start_code,
                                "submitted_code": submitted_code,
                                "test_results": case_details_log
                            }
                            temp_file_path = _maybe_save_eval_debug_record(
                                temp_eval_record,
                                stream_id=stream_id + 1,
                                turn=current_turn,
                            )
                            if temp_file_path:
                                timestamped_print(f"TEMP_LOG: Evaluation details saved to: {temp_file_path}")

                            for tm in tool_messages:
                                if tm['role'] == 'tool':
                                    try:
                                        d = json.loads(tm['content'])
                                        if d.get('status') == 'submission_received':
                                            d['status'] = comparison_msg
                                            tm['content'] = json.dumps(d)
                                    except:
                                        pass

                            if gen_verify_enabled and is_correct == 3:
                                if "No output detected" in comparison_msg:
                                    verify_user_content = (
                                        "Submission failed because no output was detected. "
                                        "Please print the final answer explicitly in solve()."
                                    )
                                else:
                                    submission_status = {
                                        "status": comparison_msg,
                                        "submitted_code": submitted_code,
                                    }
                                    try:
                                        verify_result = call_api_model_verify(
                                            problem=question,
                                            submission_status=submission_status,
                                        )
                                        if verify_result.get("skipped"):
                                            timestamped_print(
                                                f"Stream {stream_id + 1}: API verify skipped for this turn ({verify_result.get('skip_reason', 'unknown')}).",
                                                "INFO",
                                            )
                                        else:
                                            verify_user_content = (verify_result.get("content") or "").strip()
                                            if verify_user_content:
                                                timestamped_print(
                                                    f"Stream {stream_id + 1}: Appended verify feedback from API_MODEL ({verify_result.get('api_model', 'unknown')}).",
                                                    "INFO",
                                                )
                                    except Exception as verify_err:
                                        timestamped_print(
                                            f"Stream {stream_id + 1}: API verify skipped due to error: {verify_err}",
                                            "WARNING",
                                        )

                    messages.extend(tool_messages)
                    if _should_append_post_submit_assistant_ack(
                        model_path=args.model_path,
                        is_submission=is_submission,
                        next_user_content=verify_user_content,
                    ):
                        messages.append({"role": "assistant", "content": "I have submitted"})
                    if verify_user_content:
                        messages.append({"role": "user", "content": verify_user_content})
                else:
                    if args.final_test:
                        no_submission_feedback = "No submission detected. Please use submit_final_answer."
                        messages.extend([{"role": "user", "content": no_submission_feedback}])
                        temp_assistant_prefix = _build_post_no_submission_assistant_prefix(
                            model_path=args.model_path,
                            next_user_content=no_submission_feedback,
                        )
                    else:
                        end_flag = True

                current_turn += 1
                if end_flag:
                    break

            # Final Status
            if is_correct == 1:
                final_status = "correct"
            elif is_correct == -1:
                final_status = "max_tokens_reached"
            elif is_correct == 2:
                final_status = "submitted_no_execution"
            elif is_correct == 3:
                final_status = "submitted_incorrect"
            elif is_correct == 4:
                final_status = "empty_assistant_response"
            else:
                final_status = "max_turn_reached"

            all_response_streams.append({
                'stream_id': stream_id + 1,
                'final_status': final_status,
                'final_correctness': 1 if is_correct == 1 else 0,
                'total_completion_tokens': total_completion_tokens,
                'final_message_history': messages
            })
            timestamped_print(f"--- Stream {stream_id + 1} finished: {final_status} ---")

            if stop_all_streams:
                break

        result_data = _build_result_data(
            args=args,
            question=question,
            loaded_test_cases=loaded_test_cases,
            all_response_streams=all_response_streams,
            final_context=llm_service.build_prompt(messages=messages, tools=all_tools, disable_thinking=args.disable_thinking),
            termination_reason=termination_reason,
        )

        save_json(result_data, args.output_filepath)
        timestamped_print(f"Completed. Results saved to {args.output_filepath}")
    except Exception as e:
        # @codex: Persist partial progress on any exception with explicit source location.
        error_info = _extract_error_info(e)
        if current_stream_id > 0:
            error_info["stream_id"] = current_stream_id
        if current_turn > 0:
            error_info["turn"] = current_turn

        safe_final_context = ""
        if llm_service is not None and messages:
            try:
                safe_final_context = llm_service.build_prompt(messages=messages, tools=all_tools, disable_thinking=args.disable_thinking)
            except Exception as ctx_err:
                error_info["final_context_error"] = {
                    "type": type(ctx_err).__name__,
                    "message": str(ctx_err),
                }

        try:
            fallback_result_data = _build_result_data(
                args=args,
                question=question,
                loaded_test_cases=loaded_test_cases,
                all_response_streams=all_response_streams,
                final_context=safe_final_context,
                termination_reason=termination_reason,
                error_info=error_info,
            )
            save_json(fallback_result_data, args.output_filepath)
            timestamped_print(f"Saved partial result with error info to {args.output_filepath}", "WARNING")
        except Exception as save_err:
            timestamped_print(f"Failed to save partial result after exception: {save_err}", "ERROR")

        raise
