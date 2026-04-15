"""
TTT-Discover: Test-Time Training for Algorithm (Re)Discovery

Based on "Learning to Discover at Test Time" paper.
Uses Entropic RL to let LLMs rediscover algorithms at test time.
Each problem is trained independently from the initial model weights.

Architecture (vllm_server mode):
  - vLLM server: multi-GPU tensor-parallel inference (sampling)
  - Training process: single-GPU policy gradient updates
  - vLLM is stopped during training, restarted with updated weights after.

Usage:
    python -m ttt_discover.ttt_discover --config ttt_discover/configs/dijkstra.yaml
"""

import argparse
import gc
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from project_env import expand_env_vars, load_repo_env, require_repo_env_key

load_repo_env(REPO_ROOT)
require_repo_env_key("PROJECT_ROOT", REPO_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import yaml

from .data.problem_loader import load_problems
from .entropic_rl.trainer import EntropicRLStep
from .inference.vllm_client import InferenceClient
from .utils.logging_utils import MetricsLogger, StepMetrics, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """Configuration loaded from a YAML file with optional CLI overrides."""

    def __init__(self, config_path: str = None, **overrides):
        # ----- Model -----
        self.model_path: str = ""
        self.tokenizer_path: str = ""
        self.project_root: str = "."
        self.problem_dir: str = "datasets/final_test/graph-sp-dijkstra"
        self.levels: list = ["level0", "level1", "level2"]

        # ----- Inference -----
        self.inference_mode: str = "vllm_server"
        self.vllm_server_url: str = "http://localhost:8000"
        self.vllm_tensor_parallel_size: int = 4
        self.vllm_gpu_memory_utilization: float = 0.9
        self.vllm_max_model_len: int = 32768
        self.vllm_max_num_seqs: int = 64
        self.num_gpus: int = 4

        # ----- Entropic RL (Paper Table 9) -----
        self.num_samples_per_step: int = 64
        self.kl_budget: float = math.log(2)
        self.beta_search_tol: float = 0.01
        self.beta_min: float = 0.1
        self.beta_max: float = 100.0
        self.loo_epsilon: float = 1e-8
        self.clip_ratio: float = 0.2
        self.learning_rate: float = 1e-5
        self.num_inner_epochs: int = 1
        self.mini_batch_size: int = 64
        self.max_grad_norm: float = 1.0
        self.kl_coeff: float = 0.01

        # ----- Training / vLLM sync -----
        self.sync_freq: int = 5
        self.training_device: str = "cuda:0"
        self.vllm_weight_sync_mode: str = "auto"

        # ----- Per-problem training -----
        self.max_steps_per_problem: int = 30

        # ----- Legacy (kept for YAML compat) -----
        self.puct_c: float = 1.4
        self.puct_q_mode: str = "max"
        self.max_archive_size: int = 200
        self.total_steps: int = 50

        # ----- Sampling (Paper Table 9) -----
        self.temperature: float = 1.0
        self.max_new_tokens: int = 32768

        # ----- Execution sandbox -----
        self.execution_timeout: float = 20.0
        self.execution_memory_limit: int = 2048
        self.max_execution_workers: int = 8

        # ----- Reward mode -----
        self.reward_mode: str = "continuous"   # "binary" or "continuous"
        self.reward_time_limit: float = 0.9    # func_time <= this → reward = 1.0
        self.reward_time_ceiling: float = 1.5  # func_time >= this → reward = 0.0

        # ----- Eval & checkpointing -----
        self.eval_freq: int = 10
        self.eval_num_samples: int = 8
        self.save_freq: int = 10
        self.output_dir: str = "ttt_discover/outputs/dijkstra"
        self.resume_from: str = ""

        # ----- Load from YAML -----
        if config_path is not None:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = expand_env_vars(yaml.safe_load(f) or {})
            unknown = [k for k in yaml_config if not hasattr(self, k)]
            if unknown:
                logger.warning("Unknown config keys (ignored): %s", unknown)
            for k, v in yaml_config.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        # ----- CLI overrides -----
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                logger.warning("Override key '%s' not recognised — ignored.", k)

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


# ---------------------------------------------------------------------------
# vLLM server helper
# ---------------------------------------------------------------------------

class VLLMServerManager:
    """Manages the lifecycle of a vLLM server subprocess."""

    def __init__(self, model_path: str, tokenizer_path: str, config: Config):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config = config
        self._process: subprocess.Popen | None = None
        self._vllm_log_file = None
        self._vllm_log_path: str | None = None

    @property
    def url(self) -> str:
        return self.config.vllm_server_url

    def _build_cmd(self) -> list[str]:
        cfg = self.config
        port = self.url.split(":")[-1].rstrip("/")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(port),
            "--tensor-parallel-size", str(cfg.vllm_tensor_parallel_size),
            "--gpu-memory-utilization", str(cfg.vllm_gpu_memory_utilization),
            "--max-model-len", str(cfg.vllm_max_model_len),
            "--max-num-seqs", str(cfg.vllm_max_num_seqs),
            "--max-num-batched-tokens", str(cfg.vllm_max_model_len),
            "--disable-log-requests",
            "--served-model-name", "model",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--trust-remote-code",
        ]
        if self.tokenizer_path:
            cmd += ["--tokenizer", self.tokenizer_path]
        return cmd

    def is_ready(self) -> bool:
        """Check if the server is responding."""
        try:
            import requests
            resp = requests.get(f"{self.url}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def start(self, wait_timeout: int = 300, max_retries: int = 3) -> None:
        """Start the server if not already running, and wait until ready.

        If the server fails to start within *wait_timeout* seconds, kill it,
        clear GPU cache, and retry up to *max_retries* times.
        """
        if self.is_ready():
            logger.info("vLLM server already running at %s", self.url)
            return

        for attempt in range(1, max_retries + 1):
            cmd = self._build_cmd()
            logger.info(
                "Starting vLLM server (attempt %d/%d): %s",
                attempt, max_retries, " ".join(cmd),
            )

            # Redirect stdout/stderr to a log file instead of PIPE.
            # Using PIPE causes deadlock: vLLM's 4 TP workers fill the
            # 64KB pipe buffer, then all logging calls block → hang.
            vllm_log_path = os.path.join(
                self.config.output_dir or ".",
                "vllm_server.log",
            )
            os.makedirs(os.path.dirname(vllm_log_path) or ".", exist_ok=True)
            self._vllm_log_path = vllm_log_path
            self._vllm_log_file = open(vllm_log_path, "w")
            self._process = subprocess.Popen(
                cmd,
                stdout=self._vllm_log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # create new process group for clean kill
            )

            waited = 0
            interval = 5
            while waited < wait_timeout:
                time.sleep(interval)
                waited += interval
                if self._process.poll() is not None:
                    output = self._read_vllm_log()
                    logger.error(
                        "vLLM server exited with code %d (attempt %d/%d). Output:\n%s",
                        self._process.returncode, attempt, max_retries,
                        output[-4000:] if len(output) > 4000 else output,
                    )
                    break  # process died, go to retry
                if self.is_ready():
                    logger.info("vLLM server ready after %ds (attempt %d)", waited, attempt)
                    return
                logger.info("Waiting for vLLM server… (%ds/%ds)", waited, wait_timeout)

            # If we get here, server didn't start in time or crashed
            output = self._read_vllm_log()
            if output:
                logger.error(
                    "vLLM server output (timeout, attempt %d/%d):\n%s",
                    attempt, max_retries,
                    output[-4000:] if len(output) > 4000 else output,
                )

            logger.warning(
                "vLLM server failed on attempt %d/%d. Cleaning up …",
                attempt, max_retries,
            )
            self.stop()
            import torch as _torch
            gc.collect()
            _torch.cuda.empty_cache()
            if hasattr(_torch.cuda, "ipc_collect"):
                _torch.cuda.ipc_collect()
            time.sleep(8)  # longer pause before retry to let OS reclaim resources

        raise RuntimeError(
            f"vLLM server failed to start after {max_retries} attempts "
            f"(each waited up to {wait_timeout}s)"
        )

    def stop(self) -> None:
        """Terminate the server (managed or external).

        Uses process-group kill to ensure all child processes (e.g. vLLM
        tensor-parallel workers) are terminated, not just the parent.
        Then cleans up shared memory segments left by vLLM/NCCL.
        """
        if self._process is not None and self._process.poll() is None:
            pid = self._process.pid
            logger.info("Stopping managed vLLM server (pid %d) and its process group…", pid)

            # Kill the entire process group (parent + all TP worker children)
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError) as e:
                logger.debug("killpg SIGTERM failed: %s, falling back to terminate()", e)
                self._process.terminate()

            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                # Force kill the entire process group
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    self._process.kill()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

            self._process = None

        # Close the log file
        if self._vllm_log_file is not None:
            try:
                self._vllm_log_file.close()
            except Exception:
                pass
            self._vllm_log_file = None

        # Also kill any orphaned vllm workers that might be lingering
        self._kill_orphan_vllm_processes()

        # Clean up shared memory segments left by vLLM/NCCL
        self._cleanup_shared_memory()

        logger.info("vLLM server stopped and resources cleaned up.")

    @staticmethod
    def _kill_orphan_vllm_processes() -> None:
        """Kill any orphaned vllm worker processes that survived process group kill."""
        try:
            # Find processes with 'vllm' in their command line
            result = subprocess.run(
                ["pgrep", "-f", "vllm.entrypoints"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            if pids and pids[0]:
                logger.info("Killing %d orphaned vLLM processes: %s", len(pids), pids)
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except (ProcessLookupError, ValueError, PermissionError):
                        pass
                # Also kill any VllmWorker processes
                result2 = subprocess.run(
                    ["pgrep", "-f", "VllmWorker"],
                    capture_output=True, text=True, timeout=5,
                )
                pids2 = result2.stdout.strip().split()
                if pids2 and pids2[0]:
                    for pid in pids2:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except (ProcessLookupError, ValueError, PermissionError):
                            pass
                time.sleep(2)
        except Exception as e:
            logger.debug("Orphan cleanup: %s", e)

    @staticmethod
    def _cleanup_shared_memory() -> None:
        """Remove stale POSIX shared memory segments left by vLLM/NCCL.

        vLLM creates /dev/shm/psm_* files for inter-process communication.
        If workers are killed abruptly these files linger and can cause the
        next vLLM startup to hang during NCCL initialization.
        """
        try:
            shm_dir = "/dev/shm"
            if not os.path.isdir(shm_dir):
                return
            cleaned = 0
            for name in os.listdir(shm_dir):
                # vLLM shared memory segments: psm_*, nccl_*, vllm_*
                # Also CUDA MPS / IPC segments and torch shared memory
                if name.startswith(("psm_", "nccl_", "vllm_", "cuda_", "torch_")):
                    path = os.path.join(shm_dir, name)
                    try:
                        os.remove(path)
                        cleaned += 1
                    except (PermissionError, OSError):
                        pass
            if cleaned > 0:
                logger.info("Cleaned %d stale shared memory segments from %s", cleaned, shm_dir)
        except Exception as e:
            logger.debug("Shared memory cleanup: %s", e)

    def _read_vllm_log(self) -> str:
        """Read the vLLM server log file contents for debugging."""
        if self._vllm_log_path and os.path.exists(self._vllm_log_path):
            try:
                # Flush any buffered output first
                if self._vllm_log_file is not None:
                    try:
                        self._vllm_log_file.flush()
                    except Exception:
                        pass
                with open(self._vllm_log_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception:
                return ""
        return ""

    def restart_with_new_weights(self, new_model_path: str) -> None:
        """Stop server, point it at new weights, restart."""
        logger.info("Restarting vLLM with updated weights from %s", new_model_path)
        self.stop()
        self.model_path = new_model_path
        self.start()


# ---------------------------------------------------------------------------
# PUCT Reuse Buffer (Paper A.2 — per-problem state archive)
# ---------------------------------------------------------------------------

@dataclass
class SolutionState:
    """A single discovered solution state in the reuse buffer."""
    code: str                       # solve function source
    reward: float                   # R(s): best reward from this state
    visit_count: int = 0            # n(s): how many times expanded as parent
    max_child_reward: float = 0.0   # m(s): best reward among descendants
    parent: Optional["SolutionState"] = field(default=None, repr=False, compare=False)  # pointer to parent state for ancestor backprop

    @property
    def Q(self) -> float:
        """Q(s) = m(s) if expanded before, else R(s)."""
        if self.visit_count > 0:
            return self.max_child_reward
        return self.reward


class PUCTReuseBuffer:
    """Per-problem archive of solution states with PUCT-based selection.

    Implements Appendix A.2 of the paper:
    - score(s) = Q(s) + c · scale · P(s) · sqrt(1+T) / (1+n(s))
    - Q(s) = max child reward (not mean)
    - P(s) = linear rank-based prior
    - After expanding, keep top-2 children per parent, retain top-1000 globally
    """

    def __init__(self, c: float = 1.0, max_size: int = 1000):
        self.c = c
        self.max_size = max_size
        self.states: List[SolutionState] = []
        self.total_expansions: int = 0  # T

    def add_initial_empty_state(self) -> None:
        """Initialize buffer with the empty (trivial) solution."""
        self.states = [SolutionState(code="", reward=0.0)]
        self.total_expansions = 0

    def add_state(self, code: str, reward: float, parent: Optional[SolutionState] = None) -> None:
        """Add a newly discovered state to the buffer."""
        # Avoid duplicates (same code)
        for s in self.states:
            if s.code == code:
                s.reward = max(s.reward, reward)
                return
        self.states.append(SolutionState(code=code, reward=reward, parent=parent))
        self._enforce_capacity()

    def select(self) -> SolutionState:
        """Select the next state to expand using the PUCT rule.

        score(s) = Q(s) + c · scale · P(s) · sqrt(1+T) / (1+n(s))

        Returns the selected state.
        """
        if len(self.states) == 0:
            return SolutionState(code="", reward=0.0)
        if len(self.states) == 1:
            return self.states[0]

        T = self.total_expansions
        rewards = [s.reward for s in self.states]
        r_max, r_min = max(rewards), min(rewards)
        scale = r_max - r_min if r_max > r_min else 1.0

        # Rank-based prior P(s): sorted by descending reward, rank 0 = best
        sorted_indices = sorted(range(len(self.states)),
                                key=lambda i: self.states[i].reward, reverse=True)
        ranks = [0] * len(self.states)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank

        n_states = len(self.states)
        # P(s) = (|H| - rank(s)) / sum_s'(|H| - rank(s'))
        rank_values = [n_states - ranks[i] for i in range(n_states)]
        rank_sum = sum(rank_values)

        best_score = -float("inf")
        best_state = self.states[0]

        for i, state in enumerate(self.states):
            Q = state.Q
            P = rank_values[i] / rank_sum if rank_sum > 0 else 1.0 / n_states
            exploration = self.c * scale * P * math.sqrt(1 + T) / (1 + state.visit_count)
            score = Q + exploration

            if score > best_score:
                best_score = score
                best_state = state

        return best_state

    def update_after_expansion(
        self,
        parent: SolutionState,
        child_codes: List[str],
        child_rewards: List[float],
        top_k: int = 2,
    ) -> None:
        """Update buffer after expanding a parent state.

        - Increment parent visit count and total expansions
        - Backpropagate visit counts to all ancestors: n(a) += 1 for a ∈ Anc(parent)
        - Update m(parent) with best child reward
        - Keep top-k children per parent (with parent pointer set)
        - Enforce global capacity
        """
        parent.visit_count += 1
        self.total_expansions += 1

        # Backpropagate visit count to all ancestors (paper: n(a) ← n(a)+1
        # for all a ∈ {parent} ∪ Anc(parent); parent already incremented above)
        ancestor = parent.parent
        while ancestor is not None:
            ancestor.visit_count += 1
            ancestor = ancestor.parent

        if not child_codes:
            return

        # Keep top-k children by reward
        paired = sorted(zip(child_rewards, child_codes), reverse=True)
        for reward, code in paired[:top_k]:
            if reward > 0 and code:
                self.add_state(code, reward, parent=parent)

        # Update parent's max_child_reward
        best_child_r = max(child_rewards) if child_rewards else 0.0
        if best_child_r > parent.max_child_reward:
            parent.max_child_reward = best_child_r

        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        """Keep buffer within max_size by removing lowest-reward states.

        Always keep the initial empty state (if present).
        """
        if len(self.states) <= self.max_size:
            return

        # Sort by reward descending, but always keep the empty state
        empty = [s for s in self.states if s.code == ""]
        non_empty = [s for s in self.states if s.code != ""]
        non_empty.sort(key=lambda s: s.reward, reverse=True)
        self.states = empty + non_empty[:self.max_size - len(empty)]

    def summary(self) -> str:
        """Return a short summary string for logging."""
        if not self.states:
            return "empty"
        rewards = [s.reward for s in self.states]
        return (
            f"{len(self.states)} states, "
            f"best={max(rewards):.4f}, "
            f"T={self.total_expansions}"
        )


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class TTTDiscover:
    """Per-problem TTT-Discover training.

    For each problem:
      1. Reset model to initial weights
      2. Run up to ``max_steps_per_problem`` steps of Entropic RL
      3. Stop early if reward reaches 1.0 (solved)
      4. Record result

    After all problems, aggregate and report pass@1 / avg_reward.
    """

    def __init__(self, config: Config):
        self.config = config
        self._interrupted = False

        # ── Output directories ────────────────────────────────────────────
        self.ckpt_dir = os.path.join(config.output_dir, "checkpoints")
        self.logs_dir = os.path.join(config.output_dir, "logs")
        for d in (self.ckpt_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)

        # ── Logging ───────────────────────────────────────────────────────
        setup_logging(self.logs_dir)
        logger.info("Config:\n%s", json.dumps(config.to_dict(), indent=2, default=str))

        # ── Resolve model path ────────────────────────────────────────────
        if not config.model_path:
            raise ValueError("model_path must be set.")

        self._model_load_path = config.model_path
        for candidate in [
            os.path.join(config.model_path, "actor", "huggingface"),
            os.path.join(config.model_path, "huggingface"),
        ]:
            if os.path.isdir(candidate):
                logger.info("Detected verl checkpoint — loading from %s", candidate)
                self._model_load_path = candidate
                break

        # ── Tokenizer ─────────────────────────────────────────────────────
        from transformers import AutoTokenizer
        tokenizer_source = config.tokenizer_path or self._model_load_path
        logger.info("Loading tokenizer from %s …", tokenizer_source)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── vLLM server (for inference) ───────────────────────────────────
        self.vllm_mgr: VLLMServerManager | None = None
        if config.inference_mode == "vllm_server":
            self.vllm_mgr = VLLMServerManager(
                model_path=self._model_load_path,
                tokenizer_path=config.tokenizer_path,
                config=config,
            )
            self.vllm_mgr.start()

        # ── Training model ─────────────────────────────────────────────
        from transformers import AutoModelForCausalLM
        logger.info("Loading training model from %s …", self._model_load_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_load_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self._train_device = config.training_device if torch.cuda.is_available() else "cpu"

        if config.inference_mode == "vllm_server":
            self.model = self.model.to("cpu")
            logger.info(
                "Training model loaded on CPU (will move to %s during training phases).",
                self._train_device,
            )
        else:
            self.model = self.model.to(self._train_device)
            logger.info("Training model on %s (local mode).", self._train_device)

        # ── Initial weights snapshot (for per-problem reset) ──────────────
        logger.info("Saving initial weights to CPU …")
        self._initial_state_dict_cpu = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }

        # ── Reference model (for KL regularization) ──────────────────────
        if config.kl_coeff > 0:
            logger.info(
                "KL regularization enabled (kl_coeff=%.4f). "
                "Loading frozen reference model on CPU …",
                config.kl_coeff,
            )
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self._model_load_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            logger.info("Reference model loaded and frozen on CPU.")
        else:
            self.ref_model = None
            logger.info("KL regularization disabled (kl_coeff=0). No ref model.")

        # ── Optimiser ─────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )

        # ── Inference client ──────────────────────────────────────────────
        self.inference_client = InferenceClient(
            mode=config.inference_mode,
            model=self.model,
            tokenizer=self.tokenizer,
            server_url=config.vllm_server_url,
            model_path=config.tokenizer_path or self._model_load_path,
        )

        # ── Problems ──────────────────────────────────────────────────────
        logger.info("Loading problems from %s …", config.problem_dir)
        self.problems = load_problems(
            config.project_root, config.problem_dir, config.levels
        )
        if not self.problems:
            raise RuntimeError(f"No problems loaded from {config.problem_dir}")
        logger.info("Loaded %d problems.", len(self.problems))

        # ── Components ────────────────────────────────────────────────────
        self.rl_step = EntropicRLStep(config)
        self.metrics_logger = MetricsLogger(self.logs_dir)

        # ── Per-problem results ───────────────────────────────────────────
        self.problem_results: dict = {}

        # ── Response saving directory ────────────────────────────────────
        self.responses_dir = os.path.join(config.output_dir, "responses")
        os.makedirs(self.responses_dir, exist_ok=True)

        logger.info("TTT-Discover initialised successfully!")

    # ------------------------------------------------------------------
    # Save model responses to disk
    # ------------------------------------------------------------------

    def _save_step_responses(self, prob_id: str, step: int, result) -> None:
        """Save all model responses from one RL step to a JSON file.

        Saved structure per file:
            {
                "problem_id": "...",
                "step": 0,
                "num_samples": 64,
                "best_reward": 0.85,
                "mean_reward": 0.42,
                "rewards": [...],
                "samples": [
                    {
                        "index": 0,
                        "reward": 0.85,
                        "response": "...<full model output>...",
                        "solve_code": "def solve(...):\n  ..."
                    },
                    ...
                ]
            }
        """
        # Create per-problem subdirectory
        # prob_id might be like "math-strassen/level0/3" → flatten to safe filename
        safe_prob_id = prob_id.replace("/", "_").replace("\\", "_")
        prob_dir = os.path.join(self.responses_dir, safe_prob_id)
        os.makedirs(prob_dir, exist_ok=True)

        responses = result.responses or []
        solve_codes = result.solve_codes or []
        rewards = result.rewards or []

        samples = []
        for i in range(len(responses)):
            samples.append({
                "index": i,
                "reward": rewards[i] if i < len(rewards) else None,
                "response": responses[i] if i < len(responses) else "",
                "solve_code": solve_codes[i] if i < len(solve_codes) else "",
            })

        data = {
            "problem_id": prob_id,
            "step": step,
            "num_samples": result.num_samples,
            "best_reward": result.best_reward,
            "mean_reward": result.mean_reward,
            "rewards": rewards,
            "samples": samples,
        }

        filepath = os.path.join(prob_dir, f"step_{step:03d}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug("Saved %d responses to %s", len(responses), filepath)
        except Exception:
            logger.warning("Failed to save responses to %s", filepath, exc_info=True)

    # ------------------------------------------------------------------
    # Model reset (per-problem)
    # ------------------------------------------------------------------

    def _reset_model(self) -> None:
        """Restore model, optimizer, and vLLM to initial state (before each problem)."""
        self.model.load_state_dict(self._initial_state_dict_cpu, strict=True)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Reset vLLM to serve the initial weights (not the previous problem's)
        if self.vllm_mgr is not None:
            if self.vllm_mgr.model_path != self._model_load_path:
                logger.info(
                    "Resetting vLLM to initial weights (%s) …",
                    self._model_load_path,
                )
                self.vllm_mgr.stop()
                self.vllm_mgr.model_path = self._model_load_path
                torch.cuda.empty_cache()
                self.vllm_mgr.start()

        logger.info("Model and optimizer reset to initial weights.")

    # ------------------------------------------------------------------
    # GPU offloading helpers (verl-style)
    # ------------------------------------------------------------------

    def _models_to_gpu(self) -> None:
        """Stop vLLM to free GPU, then move training models to GPU."""
        if self.config.inference_mode != "vllm_server":
            return

        if self.vllm_mgr is not None:
            logger.info("Stopping vLLM server to free GPU for training …")
            self.vllm_mgr.stop()
            torch.cuda.empty_cache()

        device = self._train_device
        logger.info("Moving training model to %s …", device)
        self.model.to(device)
        if self.ref_model is not None:
            logger.info("Moving ref model to %s …", device)
            self.ref_model.to(device)
        self.inference_client.update_model(self.model)

    def _models_to_cpu(self) -> None:
        """Move training models back to CPU, then restart vLLM with updated weights."""
        if self.config.inference_mode != "vllm_server":
            return

        # Save updated weights so vLLM can load them
        if self.vllm_mgr is not None:
            sync_path = os.path.join(self.ckpt_dir, "_vllm_sync")
            os.makedirs(sync_path, exist_ok=True)
            logger.info("Saving updated weights to %s …", sync_path)
            self.model.save_pretrained(sync_path)
            self.tokenizer.save_pretrained(sync_path)
            self.vllm_mgr.model_path = sync_path

        logger.info("Moving training model back to CPU …")
        self.model.to("cpu")
        if self.ref_model is not None:
            logger.info("Moving ref model back to CPU …")
            self.ref_model.to("cpu")

        # Thorough GPU memory cleanup before restarting vLLM
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        time.sleep(3)  # give OS time to reclaim GPU memory

        if self.vllm_mgr is not None:
            logger.info("Restarting vLLM server with updated weights …")
            self.vllm_mgr.start()

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        def _handler(signum, frame):
            if self._interrupted:
                logger.warning("Forced exit.")
                self._cleanup()
                sys.exit(1)
            logger.warning("Interrupt received — finishing current problem then saving…")
            self._interrupted = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _cleanup(self):
        """Clean up resources."""
        if self.vllm_mgr is not None:
            self.vllm_mgr.stop()

    # ------------------------------------------------------------------
    # Main training loop — per-problem independent training
    # ------------------------------------------------------------------

    def run(self) -> None:
        config = self.config
        self._register_signal_handlers()

        total_problems = len(self.problems)
        max_steps = config.max_steps_per_problem

        logger.info(
            "Starting TTT-Discover: %d problems, %d steps/problem",
            total_problems, max_steps,
        )

        try:
            for prob_idx, problem in enumerate(self.problems):
                if self._interrupted:
                    logger.info("Interrupted — saving partial results.")
                    break

                prob_id = problem.problem_id
                logger.info(
                    "\n%s\n=== Problem %d / %d: %s ===\n%s",
                    "=" * 70, prob_idx + 1, total_problems, prob_id, "=" * 70,
                )

                # Reset model to initial weights for this problem
                self._reset_model()

                # Initialize per-problem PUCT reuse buffer (Paper A.2)
                reuse_buffer = PUCTReuseBuffer(
                    c=config.puct_c,
                    max_size=config.max_archive_size,
                )
                reuse_buffer.add_initial_empty_state()

                best_reward = 0.0
                best_code = ""
                prob_start_time = time.time()
                steps_done = 0

                for step in range(max_steps):
                    if self._interrupted:
                        break

                    # PUCT selects which state (previous solution) to use as prompt
                    selected_state = reuse_buffer.select()
                    problem._current_best_code = selected_state.code
                    problem._current_best_reward = selected_state.reward

                    logger.info(
                        "[%s] Step %d: PUCT selected state with reward=%.4f "
                        "(buffer: %s)",
                        prob_id, step + 1, selected_state.reward,
                        reuse_buffer.summary(),
                    )

                    try:
                        result = self.rl_step.step(
                            problem=problem,
                            model=self.model,
                            ref_model=self.ref_model,
                            tokenizer=self.tokenizer,
                            inference_client=self.inference_client,
                            optimizer=self.optimizer,
                            gpu_offload_fn=(self._models_to_gpu, self._models_to_cpu),
                        )
                    except Exception:
                        logger.error(
                            "Error at problem %s step %d:\n%s",
                            prob_id, step, traceback.format_exc(),
                        )
                        # If vLLM is down, try to recover before continuing
                        if self.vllm_mgr is not None and not self.vllm_mgr.is_ready():
                            logger.warning(
                                "vLLM server is down after error. "
                                "Attempting recovery …"
                            )
                            try:
                                # Ensure models are on CPU
                                self.model.to("cpu")
                                if self.ref_model is not None:
                                    self.ref_model.to("cpu")
                                torch.cuda.empty_cache()
                                self.vllm_mgr.start()
                                logger.info("vLLM server recovered successfully.")
                            except Exception:
                                logger.error(
                                    "vLLM recovery failed:\n%s",
                                    traceback.format_exc(),
                                )
                                logger.error(
                                    "Skipping remaining steps for problem %s.",
                                    prob_id,
                                )
                                break
                        continue

                    steps_done = step + 1

                    # Save all model responses for this step
                    self._save_step_responses(prob_id, step, result)

                    # Feed results into PUCT reuse buffer
                    child_codes = result.solve_codes or []
                    child_rewards = result.rewards or []
                    reuse_buffer.update_after_expansion(
                        parent=selected_state,
                        child_codes=child_codes,
                        child_rewards=child_rewards,
                        top_k=2,  # paper: keep top-2 children per parent
                    )

                    if result.best_reward > best_reward:
                        best_reward = result.best_reward
                        best_code = result.best_code

                    logger.info(
                        "[%s] Step %d/%d | reward=%.4f | best=%.4f | "
                        "beta=%.3f | loss=%.4f",
                        prob_id, step + 1, max_steps,
                        result.best_reward, best_reward,
                        result.beta, result.policy_loss,
                    )

                    # Early termination: solved!
                    if best_reward >= 1.0:
                        elapsed = time.time() - prob_start_time
                        logger.info(
                            "★ Problem %s SOLVED at step %d! (%.1fs)",
                            prob_id, step + 1, elapsed,
                        )
                        break

                    # ── Collapse detection ──────────────────────────────
                    # If most responses are truncated or fail code extraction,
                    # the model is degenerating.  Stop early to avoid wasting
                    # compute and further damaging the model.
                    n_resp = len(result.responses) if result.responses else 0
                    n_trunc = getattr(result, "num_truncated", 0)
                    n_no_code = (
                        sum(1 for c in result.solve_codes if c == "")
                        if result.solve_codes else n_resp
                    )
                    trunc_rate = n_trunc / max(n_resp, 1)
                    no_code_rate = n_no_code / max(n_resp, 1)

                    if n_resp > 0 and (trunc_rate > 0.5 or no_code_rate > 0.8):
                        elapsed = time.time() - prob_start_time
                        logger.warning(
                            "⚠ Model collapse detected for %s at step %d! "
                            "truncation=%.0f%% (%d/%d), no_code=%.0f%% (%d/%d). "
                            "Stopping early and resetting model. (%.1fs elapsed)",
                            prob_id, step + 1,
                            trunc_rate * 100, n_trunc, n_resp,
                            no_code_rate * 100, n_no_code, n_resp,
                            elapsed,
                        )
                        # Reset model weights to initial before moving to next problem
                        self._reset_model()
                        break

                elapsed = time.time() - prob_start_time
                self.problem_results[prob_id] = {
                    "best_reward": best_reward,
                    "best_code": best_code,
                    "steps": steps_done,
                    "elapsed_sec": round(elapsed, 1),
                    "solved": best_reward >= 1.0,
                }

                # Progress summary
                solved_so_far = sum(
                    1 for r in self.problem_results.values() if r["solved"]
                )
                logger.info(
                    "Problem %s done: reward=%.4f, steps=%d, time=%.1fs, solved=%s  "
                    "[Progress: %d/%d solved so far]",
                    prob_id, best_reward, steps_done, elapsed,
                    best_reward >= 1.0, solved_so_far, prob_idx + 1,
                )

            # ── Final report ──────────────────────────────────────────────
            self._report_final_results()

        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Final results aggregation
    # ------------------------------------------------------------------

    def _report_final_results(self) -> None:
        results = self.problem_results
        n = len(results)
        if n == 0:
            logger.warning("No problem results to report.")
            return

        solved_count = sum(1 for r in results.values() if r["solved"])
        avg_reward = sum(r["best_reward"] for r in results.values()) / n
        avg_time = sum(r["elapsed_sec"] for r in results.values()) / n
        solved_times = [r["elapsed_sec"] for r in results.values() if r["solved"]]
        solved_steps = [r["steps"] for r in results.values() if r["solved"]]

        # Per-level breakdown
        per_level: dict = {}
        for pid, r in results.items():
            # Extract level from problem_id like "graph-sp-dijkstra/level0/3"
            parts = pid.split("/")
            level = parts[1] if len(parts) >= 2 else "unknown"
            if level not in per_level:
                per_level[level] = {"solved": 0, "total": 0, "rewards": []}
            per_level[level]["total"] += 1
            per_level[level]["rewards"].append(r["best_reward"])
            if r["solved"]:
                per_level[level]["solved"] += 1

        per_level_summary = {}
        for level, data in sorted(per_level.items()):
            per_level_summary[level] = {
                "pass_at_1": data["solved"] / data["total"] if data["total"] else 0,
                "avg_reward": sum(data["rewards"]) / len(data["rewards"]),
                "solved": data["solved"],
                "total": data["total"],
            }

        summary = {
            "total_problems": n,
            "solved": solved_count,
            "pass_at_1": solved_count / n,
            "avg_reward": round(avg_reward, 4),
            "avg_time_sec": round(avg_time, 1),
            "avg_solve_time_sec": round(
                sum(solved_times) / len(solved_times), 1
            ) if solved_times else None,
            "avg_solve_steps": round(
                sum(solved_steps) / len(solved_steps), 1
            ) if solved_steps else None,
            "per_level": per_level_summary,
            "per_problem": results,
        }

        # Save to file
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, "final_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("FINAL RESULTS")
        logger.info("=" * 70)
        logger.info("  Total problems:    %d", n)
        logger.info("  Solved:            %d / %d", solved_count, n)
        logger.info("  Pass@1:            %.4f", summary["pass_at_1"])
        logger.info("  Avg reward:        %.4f", avg_reward)
        logger.info("  Avg time/problem:  %.1fs", avg_time)
        if solved_times:
            logger.info("  Avg solve time:    %.1fs", summary["avg_solve_time_sec"])
            logger.info("  Avg solve steps:   %.1f", summary["avg_solve_steps"])
        for level, data in sorted(per_level_summary.items()):
            logger.info(
                "  %s: pass@1=%.4f  avg_reward=%.4f  (%d/%d)",
                level, data["pass_at_1"], data["avg_reward"],
                data["solved"], data["total"],
            )
        logger.info("=" * 70)
        logger.info("Results saved to %s", path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTT-Discover: Test-Time Training for Algorithm Discovery",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    override_keys = [
        "model_path", "tokenizer_path", "output_dir",
        "total_steps", "num_gpus", "resume_from",
    ]
    overrides = {
        k: getattr(args, k) for k in override_keys
        if getattr(args, k) is not None
    }

    try:
        config = Config(config_path=args.config, **overrides)
        trainer = TTTDiscover(config)
        trainer.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — exiting.")
        sys.exit(130)
    except Exception:
        logger.critical("Fatal error:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
