"""Metrics tracking and logging utilities for TTT-Discover."""

import json
import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single TTT-Discover step."""
    step: int
    problem_id: str
    reward: float
    best_reward: float
    beta: float
    mean_advantage: float
    policy_loss: float
    clip_fraction: float
    approx_kl: float
    num_samples: int
    execution_success_rate: float
    puct_score: float
    archive_best_global_reward: float
    wall_time: float
    kl_ref: float = 0.0
    timestamp: float = field(default_factory=time.time)


class MetricsLogger:
    """Log metrics to JSONL file and optionally console."""
    
    def __init__(self, output_dir: str, filename: str = "metrics.jsonl"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, filename)
        self.metrics_history: list[StepMetrics] = []
    
    def log_step(self, metrics: StepMetrics):
        """Log a step's metrics."""
        self.metrics_history.append(metrics)
        
        # Append to JSONL file
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")
        
        # Console log
        logger.info(
            f"Step {metrics.step} | problem={metrics.problem_id} | "
            f"reward={metrics.reward:.3f} | best={metrics.best_reward:.3f} | "
            f"beta={metrics.beta:.2f} | loss={metrics.policy_loss:.4f} | "
            f"global_best={metrics.archive_best_global_reward:.3f}"
        )
    
    def log_eval(self, step: int, eval_metrics: dict):
        """Log evaluation results."""
        eval_entry = {"step": step, "type": "eval", "timestamp": time.time(), **eval_metrics}
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(eval_entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Eval @ step {step}: {eval_metrics}")
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.metrics_history:
            return {}
        rewards = [m.reward for m in self.metrics_history]
        return {
            "total_steps": len(self.metrics_history),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
        }


def setup_logging(output_dir: str, level=logging.INFO):
    """Setup logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    
    # File handler
    fh = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
    
    root_logger.addHandler(ch)
    root_logger.addHandler(fh)
