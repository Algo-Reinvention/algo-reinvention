import re
import math
import io
import signal
import numpy as np
import requests
import json
import time
from collections import Counter
from copy import *
from typing import List, Dict, Any, Tuple
from utils.util import timestamped_print, cprint
from contextlib import redirect_stdout
from transformers import AutoTokenizer
from vllm.inputs import TokensPrompt


class LLM_Service:
    def __init__(self, model_path: str = None, tensor_parallel_size: int = 1,
                 server_url: str = "http://localhost:8000", use_server: bool = True):
        """
        Initialize LLM service

        Args:
            model_path: Path to the model (for tokenizer only when using server)
            tensor_parallel_size: Not used when using server
            server_url: URL of the vLLM server
            use_server: Whether to use vLLM server instead of local model
        """
        self.use_server = use_server
        self.server_url = server_url
        self.model_path = model_path

        if use_server:
            # When using server, we only need tokenizer for prompt formatting
            timestamped_print(f"Initializing LLM service with server: {server_url}", level="INFO")
            if model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                timestamped_print(f"Tokenizer loaded from {model_path}", level="INFO")
            else:
                self.tokenizer = None
                timestamped_print("No tokenizer loaded - will use simple prompt formatting", level="WARNING")

            # Test server connection
            self._test_server_connection()
        else:
            # Original vLLM local loading - Removed and replaced with error
            raise NotImplementedError("Local vLLM mode (use_server=False) is not supported. Only Server mode is implemented.")

    def _test_server_connection(self):
        """Test connection to vLLM server"""
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                timestamped_print(f"✓ Connected to vLLM server successfully", level="INFO")
                timestamped_print(f"Available models: {[model['id'] for model in models.get('data', [])]}", level="INFO")
            else:
                raise Exception(f"Server responded with status {response.status_code}")
        except Exception as e:
            timestamped_print(f"✗ Failed to connect to vLLM server: {e}", level="ERROR")
            raise Exception(f"Cannot connect to vLLM server at {self.server_url}")

    def build_prompt(self, messages: List[Dict[str, str]], disable_thinking: bool = False, tools: Any = []) -> str:
        """Build prompt from messages"""
        if self.tokenizer:
            try:
                if disable_thinking:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        tools=tools,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        tools=tools,
                        add_generation_prompt=True
                    )

                bos = self.tokenizer.bos_token
                if bos:
                    while True:
                        prompt = prompt.lstrip()
                        if prompt.startswith(bos):
                            prompt = prompt[len(bos):]
                        else:
                            break

                return prompt

            except Exception as e:
                timestamped_print(f"Failed to use chat template: {e}, falling back to simple formatting", level="WARNING")
                print(messages)
                raise ValueError("Build Prompt Fail")
        else:
            raise ValueError("Build Prompt Fail")

    def inference(self, prompt: str, sampling_params) -> List[Any]:
        """Perform inference"""
        if self.use_server:
            return self._inference_server(prompt, sampling_params)
        else:
            # Local mode inference - Removed and replaced with error
            raise NotImplementedError("Local vLLM mode (use_server=False) is not supported. Only Server mode is implemented.")

    def _inference_server(self, prompt: str, sampling_params) -> List[Dict]:
        """Perform inference using vLLM server"""
        prompt_ids = self.tokenizer.encode(prompt)
        # prompt_tokens = TokensPrompt(prompt_token_ids=prompt_ids)

        # Convert sampling_params to API request format
        request_data = {
            "model": "model",  # Use the served model name
            "prompt": prompt_ids,
            "max_tokens": getattr(sampling_params, 'max_tokens', 2048),
            "temperature": getattr(sampling_params, 'temperature', 0.7),
            "top_p": getattr(sampling_params, 'top_p', 0.9),
            "top_k": getattr(sampling_params, 'top_k', 40),
            "n": getattr(sampling_params, 'n', 1),
            "stream": False,
            "stop": getattr(sampling_params, 'stop', None),
            "return_token_ids": True
        }

        model_path_lower = (self.model_path or "").lower()
        if "ministral" in model_path_lower and "unlearned" in model_path_lower:
            # @codex: Apply small anti-repetition penalties for unlearned Ministral variants.
            request_data.update({
                "frequency_penalty": 0.01,
                "presence_penalty": 0.01,
                "repetition_penalty": 1.01
            })

        # Add logprobs if enabled
        logprobs_param = getattr(sampling_params, 'logprobs', None)
        if logprobs_param is not None and logprobs_param > 0:
            request_data["logprobs"] = logprobs_param

        try:
            # print(request_data)
            response = requests.post(
                f"{self.server_url}/v1/completions",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=None
            )

            if response.status_code != 200:
                raise Exception(f"Server error: {response.status_code} - {response.text}")

            result = response.json()

            # Convert API response to local format
            outputs = []
            for choice in result.get("choices", []):
                outputs.append({
                    "prompt_ids": choice["prompt_token_ids"],
                    "prompt": self.tokenizer.decode(choice["prompt_token_ids"]),
                    "token_ids": choice["token_ids"],
                    "text": self.tokenizer.decode(choice["token_ids"]),
                    "finish_reason": choice.get("finish_reason", "stop"),
                    "logprobs": choice.get("logprobs"),
                    "index": choice.get("index", 0)
                })

            return outputs

        except Exception as e:
            timestamped_print(f"Server inference failed: {e}", level="ERROR")
            raise

    # Removed _inference_local method

    def get_text(self, outputs) -> Tuple[List[str]]:
        """Extract text from outputs"""
        if self.use_server:
            # Server format
            texts = [output["text"] for output in outputs]
            return [texts]  # Return in same format as local
        else:
            # Local vLLM format - Removed and replaced with error
            raise NotImplementedError("Local vLLM mode (use_server=False) is not supported. Only Server mode is implemented.")

    def get_finish_reason(self, outputs) -> Tuple[List[str]]:
        """Extract finish reasons from outputs"""
        if self.use_server:
            # Server format
            reasons = [output.get("finish_reason", "stop") for output in outputs]
            return [reasons]
        else:
            # Local vLLM format - Removed and replaced with error
            raise NotImplementedError("Local vLLM mode (use_server=False) is not supported. Only Server mode is implemented.")

    def get_logprobs(self, outputs) -> Tuple[List]:
        """Extract log probabilities from outputs"""
        if self.use_server:
            # Server format - process detailed logprobs
            detailed_logprobs = []
            for output in outputs:
                logprobs_data = output.get("logprobs")
                if logprobs_data is None:
                    detailed_logprobs.append(None)
                    continue

                # Parse vLLM server logprobs format
                token_logprobs = []

                # vLLM server returns logprobs in this format:
                # {"tokens": [...], "token_logprobs": [...], "top_logprobs": [...]}
                tokens = logprobs_data.get("tokens", [])
                token_logprobs_list = logprobs_data.get("token_logprobs", [])
                top_logprobs_list = logprobs_data.get("top_logprobs", [])

                for i, token in enumerate(tokens):
                    if i < len(token_logprobs_list) and i < len(top_logprobs_list):
                        token_info = {
                            "token": token,
                            "token_logprob": token_logprobs_list[i],
                            "candidates": []
                        }

                        # Add top candidates with their logprobs
                        top_logprobs_dict = top_logprobs_list[i] or {}
                        for candidate_token, candidate_logprob in top_logprobs_dict.items():
                            token_info["candidates"].append({
                                "token": candidate_token,
                                "logprob": candidate_logprob
                            })

                        token_logprobs.append(token_info)

                detailed_logprobs.append(token_logprobs)

            return [detailed_logprobs]
        else:
            # Local vLLM format - Removed and replaced with error
            raise NotImplementedError("Local vLLM mode (use_server=False) is not supported. Only Server mode is implemented.")

    def get_entropys(self, outputs) -> Tuple[List[float]]:
        """Calculate entropy from outputs"""
        if self.use_server:
            # For server, we don't have detailed logprobs usually
            timestamped_print("Entropy calculation not available with server mode", level="WARNING")
            return [[0.0 for _ in outputs]]
        else:
            # Local vLLM format - Removed and replaced with error
            raise NotImplementedError("Local vLLM mode (use_server=False) is not supported. Only Server mode is implemented.")
