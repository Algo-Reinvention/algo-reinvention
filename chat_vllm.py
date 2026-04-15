#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_env import load_repo_env


def first_nonempty(*values: str) -> str:
    for value in values:
        value = (value or "").strip()
        if value:
            return value
    return ""


def detect_default_model_path() -> str:
    return first_nonempty(
        os.getenv("QWEN3_4B_THINKING_2507_PATH", ""),
        os.getenv("QWEN3_4B_INSTRUCT_2507_PATH", ""),
        os.getenv("MINISTRAL3_14B_REASONING_2512_PATH", ""),
        os.getenv("MINISTRAL3_8B_INSTRUCT_2512_PATH", ""),
    )


def detect_default_tensor_parallel_size() -> int:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return 1
    return max(len([part for part in visible.split(",") if part.strip()]), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive streaming chat client that loads a local model directly with vLLM."
    )
    parser.add_argument(
        "--model-path",
        default=detect_default_model_path(),
        help="Local model path. If omitted, falls back to the first available standard model path in .env.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="Optional tokenizer path. Defaults to model_path.",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt added at the beginning of the conversation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens per turn.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model length passed to vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization passed to vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=detect_default_tensor_parallel_size(),
        help="Tensor parallel size. Defaults to the number of visible GPUs.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional one-shot prompt. If omitted, starts an interactive chat loop.",
    )
    return parser.parse_args()


def import_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing transformers. Install it in the vllm chat environment first.") from exc
    return AutoTokenizer


@dataclass
class EngineBundle:
    engine: Any
    sampling_params_cls: Any
    tokenizer: Any
    backend: str
    tokens_prompt_cls: Any | None = None


def build_engine_bundle(args: argparse.Namespace) -> EngineBundle:
    if not args.model_path:
        raise RuntimeError(
            "No model path resolved. Pass --model-path explicitly or set one of the standard model paths in .env."
        )
    if not Path(args.model_path).exists():
        raise RuntimeError(f"Model path does not exist: {args.model_path}")

    AutoTokenizer = import_tokenizer()
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    try:
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError as exc:
        raise RuntimeError(
            "Missing vllm. Use the vllm conda environment or install vllm before running chat_vllm.py."
        ) from exc

    engine_args = AsyncEngineArgs(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_log_stats=True,
        enforce_eager=False,
    )

    try:
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine = AsyncLLMEngine.from_engine_args(engine_args)
        return EngineBundle(
            engine=engine,
            sampling_params_cls=SamplingParams,
            tokenizer=tokenizer,
            backend="legacy_async_engine",
        )
    except ImportError:
        pass

    try:
        from vllm.inputs import TokensPrompt
        from vllm.v1.engine.async_llm import AsyncLLM

        vllm_config = engine_args.create_engine_config()
        engine = AsyncLLM.from_vllm_config(vllm_config)
        return EngineBundle(
            engine=engine,
            sampling_params_cls=SamplingParams,
            tokenizer=tokenizer,
            backend="v1_async_llm",
            tokens_prompt_cls=TokensPrompt,
        )
    except ImportError as exc:
        raise RuntimeError(
            "This vllm version does not expose a supported local async engine API for streaming chat."
        ) from exc


def build_prompt(bundle: EngineBundle, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
    tokenizer = bundle.tokenizer
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        return prompt_text, list(prompt_ids)
    except Exception:
        lines: list[str] = []
        for message in messages:
            role = message["role"].upper()
            lines.append(f"{role}: {message['content']}")
        lines.append("ASSISTANT:")
        prompt_text = "\n\n".join(lines)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return prompt_text, list(prompt_ids)


async def stream_one_response(
    bundle: EngineBundle,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    sampling_params = bundle.sampling_params_cls(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    prompt_text, prompt_ids = build_prompt(bundle, messages)
    request_id = uuid.uuid4().hex

    if bundle.backend == "legacy_async_engine":
        generator = bundle.engine.generate(prompt_text, sampling_params, request_id)
    else:
        assert bundle.tokens_prompt_cls is not None
        prompt = bundle.tokens_prompt_cls(prompt_token_ids=prompt_ids)
        generator = bundle.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

    printed = False
    previous_text = ""

    async for output in generator:
        outputs = getattr(output, "outputs", None) or []
        if not outputs:
            continue
        current_text = outputs[0].text or ""
        delta = current_text[len(previous_text) :] if current_text.startswith(previous_text) else current_text
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
            printed = True
        previous_text = current_text

    if printed:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return previous_text


async def interactive_chat(bundle: EngineBundle, args: argparse.Namespace) -> int:
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    print(f"model_path: {args.model_path}")
    print(f"backend: {bundle.backend}")
    print("commands: /reset, /exit")
    print()

    while True:
        try:
            user_input = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/reset":
            messages = [{"role": "system", "content": args.system_prompt}] if args.system_prompt else []
            print("history cleared")
            continue

        messages.append({"role": "user", "content": user_input})
        print("assistant> ", end="", flush=True)
        assistant_text = await stream_one_response(bundle, messages, args)
        messages.append({"role": "assistant", "content": assistant_text})


async def one_shot_chat(bundle: EngineBundle, args: argparse.Namespace) -> int:
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})
    await stream_one_response(bundle, messages, args)
    return 0


async def async_main() -> int:
    load_repo_env(REPO_ROOT)
    args = parse_args()
    bundle = build_engine_bundle(args)
    if args.prompt:
        return await one_shot_chat(bundle, args)
    return await interactive_chat(bundle, args)


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
