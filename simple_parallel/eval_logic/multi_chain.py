"""multi_chain reasoning

Logs:
- 2025.12.17: current only support Qwen

"""

import sys
import os
import argparse
import json
import time
from typing import List, Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import *
from .infer_module.infer_vllm_server import LLM_Service
from vllm import SamplingParams


_cached_llm_service = None  # Global variable to cache the LLM_Service instance

def get_llm_service(model_path: str = None, tensor_parallel_size: int = 1, server_url: str = "http://localhost:8000") -> LLM_Service:
    """
    Get a singleton instance of the LLM_Service.
    If the instance is already created, return it.
    """
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


@register_processor('parse_process_args')
def parse_eval_args(remaining_args: argparse.Namespace):
    parser = argparse.ArgumentParser(description="Response Generation Args.")
    # Not used
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--tensor_parallel_size", type=int, help="Number of tensor parallelism.")

    # Generation parameters
    parser.add_argument("--num", type=int, default=1, help="Number of responses to generate")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum tokens len")
    parser.add_argument("--logprobs", type=int, default=0, help="Number of most likely tokens to return logprobs for (0 to disable)")
    parser.add_argument("--disable_thinking",action="store_true", help="Disable internal thinking process or verbose logging.")

    # Prompts
    parser.add_argument("--system_prompt", type=str,
                        default="You are a helpful assistant.",
                        help="System prompt for the model.")
    parser.add_argument("--user_prompt_template", type=str,
                        default="{question}",
                        help="User prompt template.")
    parser.add_argument("--assistant_prefix", type=str,
                        default="",
                        help="Assistant prefix.")

    # Server configuration
    parser.add_argument("--server_url", type=str, default="http://localhost:8000",
                        help="vLLM server URL")

    return parser.parse_args(remaining_args)


@register_processor('check_finish')
def check_finish(args: argparse.Namespace, save_path: str) -> bool:
    """Check if the response generation is already completed"""
    if not os.path.exists(save_path):
        return False

    try:
        record = load_json(save_path)

        # Check if we have the required number of responses
        if 'generated_responses' not in record:
            return False

        if len(record['generated_responses']) < args.num:
            timestamped_print(f"Not enough responses in {save_path}. Expected {args.num}, got {len(record['generated_responses'])}.", "WARNING")
            return False

        return True

    except Exception as e:
        timestamped_print(f"Error loading JSON file {save_path}: {e}", "ERROR")
        return False


@register_processor('process')
def process_file(args) -> None:
    """
    Process a single question file: generate responses
    """
    args = fix_cli_newlines(args)

    # Load question data
    data = load_json(args.input_filepath)

    # Get LLM service
    llm_service = get_llm_service(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        server_url=args.server_url
    )

    # Prepare prompt
    question = data.get('question', '')
    user_prompt = args.user_prompt_template.format(question=question)

    if args.system_prompt != "":
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": user_prompt}
        ]

    # Generate responses
    timestamped_print(f"Generating {args.num} responses...")
    prompt = llm_service.build_prompt(messages, args.disable_thinking)
    prompt += args.assistant_prefix

    prompt_tokens = len(llm_service.tokenizer.encode(prompt, add_special_tokens=True))
    remaining_max_tokens = args.max_tokens - prompt_tokens

    # Set up sampling parameters
    sampling_params = SamplingParams(
        n=args.num,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=remaining_max_tokens,
        logprobs=args.logprobs if args.logprobs > 0 else None,
    )

    results = llm_service.inference(prompt, sampling_params)
    responses = llm_service.get_text(results)[0]
    prompt = results[0]["prompt"]

    # Get logprobs if enabled
    logprobs_data = None
    if args.logprobs > 0:
        try:
            logprobs_data = llm_service.get_logprobs(results)[0]
            timestamped_print(f"Retrieved logprobs data for {len(logprobs_data)} responses")
        except Exception as e:
            timestamped_print(f"Failed to get logprobs: {e}", "ERROR")
            raise RuntimeError(f"Failed to get logprobs: {e}")

    # Structure the generated responses
    generated_responses = []
    for i, response in enumerate(responses):
        response_data = {
            'response_id': i + 1,
            'response': response
        }

        # Add logprobs data if available
        if logprobs_data and i < len(logprobs_data) and logprobs_data[i] is not None:
            response_data['token_logprobs'] = logprobs_data[i]

        generated_responses.append(response_data)

    # Compile final results
    result_data = {
        'question': question,
        'solution': data.get('solution', ''),
        'prompt': prompt,
        'timestamp': time.time(),
        'generation_config': {
            'num': args.num,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'max_tokens': args.max_tokens,
            'logprobs': args.logprobs if args.logprobs > 0 else None,
            'disable_thinking': args.disable_thinking
        },
        'generated_responses': generated_responses
    }

    # Save results
    save_json(result_data, args.output_filepath)
    timestamped_print(f"Response generation completed. Results saved to {args.output_filepath}")
