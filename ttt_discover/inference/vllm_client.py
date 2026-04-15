"""Unified inference client supporting both local HuggingFace generate and vLLM server modes.

This module provides a single ``InferenceClient`` class that abstracts away the
differences between running inference locally (via ``model.generate()``) and
calling a remote vLLM OpenAI-compatible server.  The two modes share a common
``sample()`` API so that upstream code (MCTS search, reward evaluation, etc.)
does not need to care about the backend.

Typical usage – local mode::

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    client = InferenceClient(mode="local", model=model, tokenizer=tokenizer)
    responses = client.sample(["Solve this problem…"], n=4)

Typical usage – vLLM server mode::

    client = InferenceClient(
        mode="vllm_server",
        model_path="deepseek-ai/deepseek-coder-7b",
        server_url="http://localhost:8000",
    )
    responses = client.sample(["Solve this problem…"], n=4)
"""

import logging
from typing import Optional

import requests
import torch

logger = logging.getLogger(__name__)


class InferenceClient:
    """Unified inference interface supporting local HF generate and vLLM server modes."""

    # Recognised mode strings ------------------------------------------------
    MODE_LOCAL = "local"
    MODE_VLLM_SERVER = "vllm_server"
    _VALID_MODES = {MODE_LOCAL, MODE_VLLM_SERVER}

    # ---------------------------------------------------------------------- #
    #  Construction / initialisation                                          #
    # ---------------------------------------------------------------------- #

    def __init__(
        self,
        mode: str = "local",
        model=None,
        tokenizer=None,
        server_url: str = "http://localhost:8000",
        model_path: Optional[str] = None,
    ):
        """Initialise the inference client.

        Args:
            mode: ``"local"`` for HF generate, ``"vllm_server"`` for vLLM server.
            model: HuggingFace model (required for local mode).
            tokenizer: HuggingFace tokenizer (required for both modes).
            server_url: vLLM server URL (for ``vllm_server`` mode).
            model_path: Model path for loading tokenizer if not provided
                (for ``vllm_server`` mode).

        Raises:
            AssertionError: If required arguments are missing for the chosen mode.
            ConnectionError: If the vLLM server cannot be reached.
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown mode {mode!r}. Choose from {sorted(self._VALID_MODES)}."
            )

        self.mode = mode
        self.model = model
        self.tokenizer = tokenizer
        self.server_url = server_url.rstrip("/")

        # Truncation tracking — populated after each sample() call.
        # List of bools, one per response (flattened across prompts).
        self.last_truncated: list[bool] = []
        # Summary counts for the last sample() call.
        self.last_num_truncated: int = 0
        self.last_num_total: int = 0

        if mode == self.MODE_LOCAL:
            assert model is not None, "model required for local mode"
            assert tokenizer is not None, "tokenizer required for local mode"
        elif mode == self.MODE_VLLM_SERVER:
            if tokenizer is None and model_path:
                from transformers import AutoTokenizer

                logger.info("Loading tokenizer from %s for vllm_server mode …", model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
            self._test_connection()

    # ---------------------------------------------------------------------- #
    #  Connection helpers                                                     #
    # ---------------------------------------------------------------------- #

    def _test_connection(self) -> None:
        """Test vLLM server connection.

        Raises:
            ConnectionError: If the server is unreachable or returns a non-200 status.
        """
        try:
            resp = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                logger.info("Connected to vLLM server at %s", self.server_url)
            else:
                raise ConnectionError(f"Server returned {resp.status_code}")
        except requests.ConnectionError as exc:
            logger.error("Failed to connect to vLLM server: %s", exc)
            raise ConnectionError(
                f"Cannot reach vLLM server at {self.server_url}"
            ) from exc
        except Exception as exc:
            logger.error("Failed to connect to vLLM server: %s", exc)
            raise

    # ---------------------------------------------------------------------- #
    #  Prompt construction                                                    #
    # ---------------------------------------------------------------------- #

    def build_prompt(
        self,
        problem_text: str,
        best_code: str = "",
        best_reward: float = 0.0,
    ) -> str:
        """Build the prompt for the model.

        Args:
            problem_text: The problem description.
            best_code: Previous best solve function (empty if first attempt).
            best_reward: Previous best reward score.

        Returns:
            Formatted prompt string ready for tokenisation.
        """
        system_msg = (
            "You are an expert Python programmer. "
            "Solve the given problem by writing a solve function."
        )

        if best_code and best_reward > 0:
            user_msg = (
                f"{problem_text}\n\n"
                f"### Previous Best Attempt (score: {best_reward:.2f}):\n"
                f"```python\n{best_code}\n```\n\n"
                "Please improve upon this solution or try a different approach."
            )
        else:
            user_msg = problem_text

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for tokenizers without a chat template
            logger.warning(
                "Tokenizer does not support apply_chat_template; using fallback format."
            )
            prompt = (
                f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"
            )

        return prompt

    # ---------------------------------------------------------------------- #
    #  Sampling – public API                                                  #
    # ---------------------------------------------------------------------- #

    def sample(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> list[list[str]]:
        """Sample *n* responses per prompt.

        Args:
            prompts: List of prompt strings.
            n: Number of samples per prompt.
            temperature: Sampling temperature (0 → greedy).
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Nested list ``responses[prompt_idx][sample_idx]`` of decoded strings.
        """
        if not prompts:
            return []

        if self.mode == self.MODE_LOCAL:
            return self._sample_local(prompts, n, temperature, max_new_tokens)
        else:
            return self._sample_vllm(prompts, n, temperature, max_new_tokens)

    # ---------------------------------------------------------------------- #
    #  Sampling – local HuggingFace backend                                   #
    # ---------------------------------------------------------------------- #

    def _sample_local(
        self,
        prompts: list[str],
        n: int,
        temperature: float,
        max_new_tokens: int,
    ) -> list[list[str]]:
        """Sample using ``HuggingFace model.generate()``.

        Each prompt is processed individually to avoid complex padding logic
        across heterogeneous prompt lengths.
        """
        results: list[list[str]] = []
        all_truncated: list[bool] = []
        self.model.eval()
        # Unwrap DataParallel for .generate() and .device access
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        device = next(raw_model.parameters()).device

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            prompt_length = input_ids.shape[1]

            with torch.no_grad():
                outputs = raw_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    num_return_sequences=n,
                    pad_token_id=(
                        self.tokenizer.pad_token_id
                        or self.tokenizer.eos_token_id
                    ),
                )

            prompt_responses: list[str] = []
            for i in range(n):
                response_ids = outputs[i][prompt_length:]
                response_text = self.tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )
                prompt_responses.append(response_text)
                # Detect truncation: if generated exactly max_new_tokens and
                # the last token is NOT eos, it was likely truncated.
                is_truncated = (
                    len(response_ids) >= max_new_tokens
                    and (
                        response_ids[-1].item() != self.tokenizer.eos_token_id
                        if len(response_ids) > 0
                        else False
                    )
                )
                all_truncated.append(is_truncated)

            results.append(prompt_responses)

        # Update truncation stats
        self.last_truncated = all_truncated
        self.last_num_truncated = sum(all_truncated)
        self.last_num_total = len(all_truncated)

        if self.last_num_truncated > 0:
            logger.warning(
                "Truncation (local): %d / %d responses hit max_new_tokens (%d).",
                self.last_num_truncated,
                self.last_num_total,
                max_new_tokens,
            )

        return results

    # ---------------------------------------------------------------------- #
    #  Sampling – vLLM server backend                                         #
    # ---------------------------------------------------------------------- #

    def _sample_vllm(
        self,
        prompts: list[str],
        n: int,
        temperature: float,
        max_new_tokens: int,
    ) -> list[list[str]]:
        """Sample using the vLLM OpenAI-compatible completions endpoint.

        Prompts are sent as token-ID lists (matching the existing
        ``infer_vllm_server.py`` pattern) so that special tokens are preserved
        exactly.

        Also populates ``self.last_truncated`` with per-response truncation
        flags (``True`` when ``finish_reason == "length"``).
        """
        results: list[list[str]] = []
        all_truncated: list[bool] = []

        for prompt in prompts:
            # Tokenize and send as token IDs to preserve special tokens
            prompt_ids = self.tokenizer.encode(prompt)

            request_data = {
                "model": "model",
                "prompt": prompt_ids,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "n": n,
                "stream": False,
            }

            try:
                response = requests.post(
                    f"{self.server_url}/v1/completions",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=None,  # generation can be long-running
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Server error: {response.status_code} - {response.text}"
                    )

                result = response.json()
                choices = result.get("choices", [])

                prompt_responses: list[str] = []
                for choice in choices:
                    prompt_responses.append(choice.get("text", ""))
                    is_truncated = choice.get("finish_reason") == "length"
                    all_truncated.append(is_truncated)

                # Pad if the server returned fewer responses than requested
                while len(prompt_responses) < n:
                    logger.warning(
                        "vLLM returned %d responses, expected %d; padding with empty strings.",
                        len(prompt_responses),
                        n,
                    )
                    prompt_responses.append("")
                    all_truncated.append(True)  # missing = treat as truncated

                results.append(prompt_responses)

            except Exception as exc:
                logger.error("vLLM inference failed for a prompt: %s", exc)
                results.append([""] * n)
                all_truncated.extend([True] * n)

        # Update truncation stats
        self.last_truncated = all_truncated
        self.last_num_truncated = sum(all_truncated)
        self.last_num_total = len(all_truncated)

        if self.last_num_truncated > 0:
            logger.warning(
                "Truncation: %d / %d responses hit max_tokens (%d). "
                "Consider increasing max_new_tokens.",
                self.last_num_truncated,
                self.last_num_total,
                max_new_tokens,
            )

        return results

    # ---------------------------------------------------------------------- #
    #  Log-probability computation (local mode only)                          #
    # ---------------------------------------------------------------------- #

    def compute_log_probs(
        self,
        prompt_texts: list[str],
        response_texts: list[str],
        no_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log probabilities for prompt + response pairs.

        Only works in **local** mode because it requires direct access to the
        model's logits.

        Args:
            prompt_texts: List of prompt strings.
            response_texts: List of response strings (same length as *prompt_texts*).
            no_grad: If True (default), run under ``torch.no_grad()`` — suitable
                for computing frozen old/ref log-probs.  Set to False when
                computing new log-probs that need gradients for PPO training.

        Returns:
            A tuple ``(padded_log_probs, masks)`` where both tensors have shape
            ``(batch, max_response_len)``.  ``masks`` is 1.0 for real tokens and
            0.0 for padding positions.

        Raises:
            AssertionError: If called in ``vllm_server`` mode.
        """
        # compute_log_probs always uses the in-process model regardless of
        # inference mode (vLLM server is only used for sampling).
        if self.model is None:
            raise RuntimeError(
                "compute_log_probs requires an in-process model, but model is None. "
                "Make sure to pass the training model when constructing InferenceClient."
            )
        assert len(prompt_texts) == len(response_texts), (
            "prompt_texts and response_texts must have the same length"
        )

        if no_grad:
            self.model.eval()
        # Handle DataParallel wrapper
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        device = next(raw_model.parameters()).device

        all_log_probs: list[torch.Tensor] = []
        response_lengths: list[int] = []

        for prompt_text, response_text in zip(prompt_texts, response_texts):
            full_text = prompt_text + response_text

            full_ids = self.tokenizer.encode(
                full_text, return_tensors="pt", add_special_tokens=False
            ).to(device)
            prompt_ids = self.tokenizer.encode(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(device)

            prompt_len = prompt_ids.shape[1]

            if no_grad:
                with torch.no_grad():
                    outputs = raw_model(full_ids)
                    logits = outputs.logits
            else:
                outputs = raw_model(full_ids)
                logits = outputs.logits

            # Shift: logits[t] predicts token[t+1]
            shift_logits = logits[:, :-1, :]
            shift_labels = full_ids[:, 1:]

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Only keep response portion (prompt_len-1 onwards because of shift)
            response_log_probs = token_log_probs[:, prompt_len - 1 :]
            all_log_probs.append(response_log_probs.squeeze(0))
            response_lengths.append(response_log_probs.shape[1])

        # Pad to same length across the batch
        max_len = max(response_lengths) if response_lengths else 0
        padded = torch.zeros(len(all_log_probs), max_len, device=device)
        masks = torch.zeros(len(all_log_probs), max_len, device=device)

        for i, lp in enumerate(all_log_probs):
            length = lp.shape[0]
            padded[i, :length] = lp
            masks[i, :length] = 1.0

        return padded, masks

    # ---------------------------------------------------------------------- #
    #  Model management                                                       #
    # ---------------------------------------------------------------------- #

    def update_model(self, model) -> None:
        """Update the model reference used for log-prob computation.

        In vllm_server mode, sampling goes via the HTTP API but
        ``compute_log_probs`` still uses the local model.  So we always
        update the reference.

        Args:
            model: The new HuggingFace model to use for subsequent inference.
        """
        self.model = model
        logger.debug("Model reference updated (mode=%s).", self.mode)
