"""
LLM-as-a-Judge Unlearn
"""

import os
import logging
import json
import requests
import datetime
import sys
from pathlib import Path
from typing import Any, Dict
from openai import OpenAI
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from project_env import load_repo_env

load_repo_env(REPO_ROOT)

logger = logging.getLogger(__name__)
MODEL_PATH = (
    os.getenv("UNLEARN_REWARD_MODEL_PATH", "").strip()
    or os.getenv("QWEN3_4B_THINKING_2507_PATH", "").strip()
    or os.getenv("QWEN3_4B_INSTRUCT_2507_PATH", "").strip()
)

def compute_score(prompt_str, solution_strs, **kwargs) -> Any:
    if not MODEL_PATH:
        raise ValueError("Reward model path is empty. Set UNLEARN_REWARD_MODEL_PATH or a base model path in .env.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    server_url = os.getenv("UNLEARN_REWARD_SERVER_URL", "").strip()
    if not server_url:
        raise ValueError("UNLEARN_REWARD_SERVER_URL is empty. Set it in the environment or .env.")
    target_str = "ijkstra"
    # prefix = '```json\n{"readable_or_not": '
    # prefix = '<think>\n\n</think>\n\n```json\n'
    # prefix = "<think>\n\n</think>\n\nLet's check step by step"
    prefix = "```json\n"
    stop_str = "```"
    reward_debug_log_dir = kwargs.get("reward_debug_log_dir", None)
    
    reward_prompts = []
    scores = []
    
    for solution_str in solution_strs:
        messages = [
            {'role': 'user', 'content': f"""Below is the user's request and the assistant's response. Please check:
Whether the assistant's response readable (completely smooth, fluent, and easy to read, with no unnatural mixing of languages)?

Please reason chunk by chunk and put the final boolean value (true/false) of `readable_or_not` in a JSON dictionary within ```json, as shown in the example below:
```json
{{"readable_or_not": true}}
```

<user_query>
{prompt_str}
</user_query>

<assistant_response>
{solution_str}
<assistant_response>"""}
        ]
        reward_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, tools=None, add_generation_prompt=True) + prefix
        )

    request_data = {
        "model": "model",
        "prompt": reward_prompts,
        "max_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 40,
        "n": 1,
        "stream": False,
        "stop": [stop_str]
    }

    try:
        response = requests.post(
            f"{server_url}/v1/completions",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=None
        )
        if response.status_code != 200:
            raise Exception(f"Server error: {response.status_code} - {response.text}")
        
        result = response.json()
        outputs = []
        for choice in result.get("choices", []):
            outputs.append({
                "text": prefix + choice["text"] + stop_str,
                "finish_reason": choice.get("finish_reason", "stop"),
                "index": choice.get("index", 0)
            })
    except Exception as e:
        print(f"Server inference failed: {e}")
        raise

    debug_log_entries = []

    for i, (output, solution_str) in enumerate(zip(outputs, solution_strs)):
        text = output["text"]
        try:
            result_dict = extract_and_parse_model_output(text)
        except Exception as e:
            result_dict = {"readable_or_not": False}
            raise ValueError("")
        
        contains_forbidden_str = target_str in solution_str
        is_readable = result_dict.get("readable_or_not", False)
        
        if contains_forbidden_str:
            score = 0.0
            reason = "failed_forbidden_string_filter"
        else:
            score = 1.0 if is_readable else 0.0
            reason = "passed_filter_based_on_readability"
        
        scores.append(score)

        debug_log_entries.append({
            "index": i,
            "solution_str": solution_str,
            "reward_model_prompt": reward_prompts[i],
            "reward_model_raw_output": text,
            "parsed_result": result_dict,
            "forbidden_str_found": contains_forbidden_str,
            "final_score": score,
            "decision_reason": reason
        })

    if reward_debug_log_dir and debug_log_entries:
        try:
            os.makedirs(reward_debug_log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_path = os.path.join(reward_debug_log_dir, f"reward_log_{timestamp}.json")
            
            log_payload = {
                "timestamp": timestamp,
                "prompt_str": prompt_str,
                "forbidden_string_pattern": target_str,
                "samples": debug_log_entries
            }
            
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_payload, f, ensure_ascii=False, indent=4)
        except Exception:
            pass
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_payload = {
            "timestamp": timestamp,
            "prompt_str": prompt_str,
            "forbidden_string_pattern": target_str,
            "samples": debug_log_entries
        }
        print(outputs)
        print(log_payload)
    
    return scores

class ExtractionError(Exception):
    pass

def extract_and_parse_model_output(text: Any) -> Dict[str, bool]:
    try:
        model_output_text = text
        if not model_output_text:
            raise ExtractionError("Model response content is empty.")
    except (AttributeError, IndexError, TypeError) as e:
        raise ExtractionError(f"Failed to extract model text output. The structure may be incorrect. Original error: {e}")
    
    try:
        start_tag = "```json"
        end_tag = "```"
        
        start_index = model_output_text.find(start_tag)
        if start_index == -1:
            raise ExtractionError(f"Start tag '{start_tag}' not found in model output.")
        
        json_start = start_index + len(start_tag)
        end_index = model_output_text.find(end_tag, json_start)
        if end_index == -1:
            raise ExtractionError(f"End tag '{end_tag}' not found in model output.")
        
        json_str = model_output_text[json_start:end_index].strip()
        if not json_str:
             raise ExtractionError("Extracted JSON string content is empty.")

    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"Failed to extract JSON block. Original error: {e}")
        
    try:
        result_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Failed to parse extracted JSON string. JSON decode error: {e}\nAttempted string:\n{json_str}")
        
    required_keys = ["readable_or_not"]
    for key in required_keys:
        if key not in result_dict:
            raise ExtractionError(f"Parsed JSON dictionary is missing required key: '{key}'.")
        if not isinstance(result_dict[key], bool):
            raise ExtractionError(f"Value for key '{key}' is not a boolean. Actual type: {type(result_dict[key])}.")
            
    return result_dict

if __name__ == "__main__":
    print("Starting LLM-as-a-Judge logic test...\n")

    test_prompt = "Explain the shortest path algorithm."
    
    test_solutions = [
        ""
    ]

    try:
        final_scores = compute_score(test_prompt, test_solutions, mock=True)
        print(f"\nFinal score list: {final_scores}")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
