from dataclasses import dataclass, field
from typing import Optional


@dataclass
class State:
    """A state in the TTT-Discover search tree.

    Each state corresponds to a problem + the best solution found so far.
    """

    problem_id: str                    # Unique identifier from Problem
    best_code: str = ""                # Best solve function code found so far
    best_reward: float = 0.0           # Best reward achieved
    visit_count: int = 0               # n(s): times this state has been selected by PUCT
    Q: float = 0.0                     # Quality value: best reward seen (backpropagated)
    total_reward_sum: float = 0.0      # Sum of all rewards for averaging

    def update(self, reward: float, code: str, q_mode: str = "max"):
        """Update state with new result from RL step.

        Args:
            reward: Reward from the latest RL step (best reward in the batch).
            code: Best code from the latest RL step.
            q_mode: How to compute Q(s). "max" uses the best reward ever seen;
                    "avg" uses the running average of all step-level rewards.
        """
        self.visit_count += 1
        self.total_reward_sum += reward
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_code = code
        # Q-value computation
        if q_mode == "avg":
            self.Q = self.total_reward_sum / self.visit_count
        else:  # "max" (default, original behaviour)
            self.Q = max(self.Q, reward)
