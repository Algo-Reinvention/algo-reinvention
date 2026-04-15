import logging
from typing import Optional

from .state import State

logger = logging.getLogger(__name__)


class Archive:
    """Maintains the archive of discovered states and their rewards.

    Per paper A.2: The archive stores states with their best rewards
    and supports PUCT-based selection.
    """

    def __init__(self, max_size: int = 200):
        self.states: dict[str, State] = {}  # problem_id -> State
        self.max_size = max_size
        self.total_visits = 0               # T: total visits across all states

    def add_state(self, problem_id: str) -> State:
        """Add a new state for a problem.

        If the state already exists, return the existing one.
        If the archive is at capacity, log a warning and refuse to add.
        """
        if problem_id in self.states:
            logger.debug("State for problem '%s' already exists, returning it.", problem_id)
            return self.states[problem_id]

        if len(self.states) >= self.max_size:
            logger.warning(
                "Archive at max capacity (%d). Cannot add state for problem '%s'.",
                self.max_size,
                problem_id,
            )
            raise ValueError(
                f"Archive is full ({self.max_size} states). "
                f"Cannot add state for problem '{problem_id}'."
            )

        state = State(problem_id=problem_id)
        self.states[problem_id] = state
        logger.info("Added new state for problem '%s'. Archive size: %d", problem_id, len(self.states))
        return state

    def get_state(self, problem_id: str) -> Optional[State]:
        """Get state by problem_id. Returns None if not found."""
        return self.states.get(problem_id)

    def update_state(self, problem_id: str, reward: float, code: str, q_mode: str = "max"):
        """Update state with new RL result and increment total visits."""
        state = self.states.get(problem_id)
        if state is None:
            raise KeyError(f"No state found for problem_id '{problem_id}'")

        old_reward = state.best_reward
        state.update(reward, code, q_mode=q_mode)
        self.total_visits += 1

        if reward > old_reward:
            logger.info(
                "Problem '%s': new best reward %.4f -> %.4f (visit #%d)",
                problem_id,
                old_reward,
                state.best_reward,
                state.visit_count,
            )
        else:
            logger.debug(
                "Problem '%s': reward %.4f (best stays %.4f, visit #%d)",
                problem_id,
                reward,
                state.best_reward,
                state.visit_count,
            )

    def get_all_states(self) -> list[State]:
        """Get all states sorted by problem_id."""
        return sorted(self.states.values(), key=lambda s: s.problem_id)

    def get_reward_scale(self) -> float:
        """Return max(r) - min(r) across all states. Used for PUCT scale.

        Returns 0.0 if there are fewer than 2 states.
        """
        if len(self.states) < 2:
            return 0.0
        rewards = [s.best_reward for s in self.states.values()]
        return max(rewards) - min(rewards)

    def get_best_global(self) -> tuple[float, str, str]:
        """Return (best_reward, best_code, problem_id) across all states.

        Raises ValueError if the archive is empty.
        """
        if not self.states:
            raise ValueError("Archive is empty — no best state available.")
        best_state = max(self.states.values(), key=lambda s: s.best_reward)
        return (best_state.best_reward, best_state.best_code, best_state.problem_id)

    def to_dict(self) -> dict:
        """Serialize archive to dict for JSON saving."""
        return {
            "max_size": self.max_size,
            "total_visits": self.total_visits,
            "states": {
                pid: {
                    "problem_id": s.problem_id,
                    "best_code": s.best_code,
                    "best_reward": s.best_reward,
                    "visit_count": s.visit_count,
                    "Q": s.Q,
                    "total_reward_sum": s.total_reward_sum,
                }
                for pid, s in self.states.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Archive":
        """Restore archive from saved dict."""
        archive = cls(max_size=data.get("max_size", 200))
        archive.total_visits = data.get("total_visits", 0)

        for pid, sdata in data.get("states", {}).items():
            state = State(
                problem_id=sdata["problem_id"],
                best_code=sdata.get("best_code", ""),
                best_reward=sdata.get("best_reward", 0.0),
                visit_count=sdata.get("visit_count", 0),
                Q=sdata.get("Q", 0.0),
                total_reward_sum=sdata.get("total_reward_sum", 0.0),
            )
            archive.states[pid] = state

        logger.info(
            "Restored archive with %d states, %d total visits.",
            len(archive.states),
            archive.total_visits,
        )
        return archive
