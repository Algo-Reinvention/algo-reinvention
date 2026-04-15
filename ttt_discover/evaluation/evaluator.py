"""Evaluation module that tests model's reinvention ability."""

import logging
import math
import torch
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from ..data.problem_loader import Problem
from ..reward.verifier import compute_verifier_reward
from ..utils.code_extraction import extract_solve_function

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate whether the model has successfully reinvented the algorithm.
    
    Uses groundtruth test cases to check code correctness.
    Computes pass@1, pass@k, and other metrics.
    """
    
    def __init__(self, inference_client, problems: list, 
                 timeout: float = 20.0, memory_limit_mb: int = 2048):
        self.inference_client = inference_client
        self.problems = problems
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
    
    def evaluate(self, num_samples: int = 8, temperature: float = 0.7,
                 max_new_tokens: int = 2048) -> dict:
        """Run full evaluation on all problems.
        
        For each problem:
        1. Sample num_samples responses
        2. Extract solve functions
        3. Execute against test cases
        4. Compute pass@1, pass@k metrics
        
        Returns dict with:
        - overall_pass_at_1: fraction of problems solved at least once
        - overall_avg_reward: average reward across all problems
        - per_level_metrics: metrics broken down by level
        - per_problem_details: list of per-problem results
        """
        all_results = []
        
        for problem in self.problems:
            prompt = self.inference_client.build_prompt(problem.problem_text)
            responses_list = self.inference_client.sample(
                [prompt], n=num_samples, temperature=temperature,
                max_new_tokens=max_new_tokens
            )
            responses = responses_list[0] if responses_list else []
            
            # Extract and evaluate
            rewards = []
            for resp in responses:
                solve_code = extract_solve_function(resp)
                if solve_code:
                    reward = compute_verifier_reward(solve_code, problem, self.timeout, self.memory_limit_mb)
                else:
                    reward = 0.0
                rewards.append(reward)
            
            # Compute metrics
            has_perfect = any(r >= 1.0 for r in rewards)
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            execution_rate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
            
            # pass@k estimation (unbiased estimator)
            pass_at_1 = 1.0 if has_perfect else 0.0
            n_correct = sum(1 for r in rewards if r >= 1.0)
            pass_at_k = compute_pass_at_k(len(rewards), n_correct, 1)
            
            all_results.append({
                "problem_id": problem.problem_id,
                "level": problem.level,
                "avg_reward": avg_reward,
                "pass_at_1": pass_at_1,
                "pass_at_k_estimated": pass_at_k,
                "n_correct": n_correct,
                "n_total": len(rewards),
                "execution_rate": execution_rate,
                "rewards": rewards,
            })
        
        # Aggregate
        overall = self._aggregate_results(all_results)
        return overall
    
    def _aggregate_results(self, results: list[dict]) -> dict:
        """Aggregate per-problem results into overall metrics."""
        if not results:
            return {"overall_pass_at_1": 0.0, "overall_avg_reward": 0.0, "per_problem_details": []}
        
        # Overall
        overall_pass = sum(1 for r in results if r["pass_at_1"] > 0) / len(results)
        overall_reward = sum(r["avg_reward"] for r in results) / len(results)
        
        # Per level
        levels = set(r["level"] for r in results)
        per_level = {}
        for level in sorted(levels):
            level_results = [r for r in results if r["level"] == level]
            per_level[level] = {
                "pass_at_1": sum(1 for r in level_results if r["pass_at_1"] > 0) / len(level_results),
                "avg_reward": sum(r["avg_reward"] for r in level_results) / len(level_results),
                "n_problems": len(level_results),
            }
        
        return {
            "overall_pass_at_1": overall_pass,
            "overall_avg_reward": overall_reward,
            "n_problems": len(results),
            "per_level_metrics": per_level,
            "per_problem_details": results,
        }


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k.
    
    n: total samples
    c: number of correct samples
    k: k value
    
    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    
    # Use log space for numerical stability
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result
