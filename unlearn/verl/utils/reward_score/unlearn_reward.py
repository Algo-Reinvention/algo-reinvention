"""
Simple Unlearn
"""

import os
import logging
import json
import sys
from pathlib import Path
from typing import Any, Dict
from openai import OpenAI
from typing import Dict, Any, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from project_env import load_repo_env, require_repo_env_key

load_repo_env(REPO_ROOT)
require_repo_env_key("PROJECT_ROOT", REPO_ROOT)

logger = logging.getLogger(__name__)

api_key = os.getenv("API_KEY", "").strip()
base_url = os.getenv("BASE_URL", "").strip()
model_name = os.getenv("API_MODEL_NAME", "").strip()

client = None
if api_key and base_url:
    client = OpenAI(api_key=api_key, base_url=base_url)

def compute_score(prompt_str, solution_str, **kwargs) -> Any:
    if client is None:
        raise RuntimeError("API_KEY and BASE_URL must be set before using this reward path.")
    if not model_name:
        raise RuntimeError("API_MODEL_NAME must be set before using this reward path.")

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'user', 'content': f"Below is the user's request and the assistant's response. Please check:\n1. Is the assistant's response readable (not garbled text or difficult-to-read words)?\n2. Has the assistant almost forgotten the relevant knowledge?\n\nPlease output only the boolean value of `readable_or_not` and `forget_or_not` in a JSON dictionary within ```json, as shown in the example below:\n```json\n{{\"readable_or_not\": true, \"forget_or_not\": true}}\n```\n\n<user_query>\n{prompt_str}\n</user_query>\n\n<assistant_response>\n{solution_str}\n<assistant_response>"}
        ]
    )

    result_dict = extract_and_parse_model_output(completion)

    score = 1.0 if result_dict["forget_or_not"] == 0 and result_dict["readable_or_not"] else 0.0
    
    return score

class ExtractionError(Exception):
    pass

def extract_and_parse_model_output(completion: Any) -> Dict[str, bool]:
    try:
        model_output_text = completion.choices[0].message.content
        if not model_output_text:
            raise ExtractionError("Model response content is empty.")
    except (AttributeError, IndexError, TypeError) as e:
        raise ExtractionError(f"Failed to extract model text output. The structure may be invalid. Original error: {e}")
    
    try:
        start_tag = "```json"
        end_tag = "```"
        
        start_index = model_output_text.find(start_tag)
        if start_index == -1:
            raise ExtractionError(f"Start tag '{start_tag}' was not found in the model output.")
        
        json_start = start_index + len(start_tag)
        end_index = model_output_text.find(end_tag, json_start)
        if end_index == -1:
            raise ExtractionError(f"End tag '{end_tag}' was not found in the model output.")
        
        json_str = model_output_text[json_start:end_index].strip()
        if not json_str:
             raise ExtractionError("The extracted JSON string is empty.")

    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"Failed to extract the JSON block. Original error: {e}")
        
    try:
        result_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Failed to parse the extracted JSON string. JSON error: {e}\nAttempted string:\n{json_str}")
        
    required_keys = ["readable_or_not", "forget_or_not"]
    for key in required_keys:
        if key not in result_dict:
            raise ExtractionError(f"The parsed JSON dictionary is missing required key: '{key}'.")
        if not isinstance(result_dict[key], bool):
            raise ExtractionError(f"Value for key '{key}' is not a boolean. Actual type: {type(result_dict[key])}.")
            
    return result_dict
