import logging
import math
import random
from typing import Optional

from .archive import Archive
from .state import State

logger = logging.getLogger(__name__)


class PUCTSelector:
    """PUCT-based state selection (paper Appendix A.2).

    Uses the formula:
        score(s) = Q(s) + c · scale · P(s) · √(1 + T) / (1 + n(s))

    where:
        Q(s)   = best reward seen for state s
        c      = exploration constant (default 1.4)
        scale  = max(r) - min(r) across archive (with minimum of 0.01)
        P(s)   = rank-based linear prior (higher rank = higher prior)
        T      = total visits across all states
        n(s)   = visit count for state s
    """

    SCALE_FLOOR = 0.01  # Minimum scale to avoid zero exploration bonus

    def __init__(self, c: float = 1.4):
        self.c = c

    def compute_prior(self, states: list[State]) -> dict[str, float]:
        """Compute rank-based linear prior P(s).

        States are ranked by Q value in *ascending* order so that states
        with LOWER Q receive HIGHER rank numbers and therefore HIGHER
        prior — this encourages exploration of under-explored problems.

        P(s) = rank(s) / sum_of_all_ranks

        Ranks are 1-based: the state with the lowest Q gets rank N
        (highest prior), and the state with the highest Q gets rank 1
        (lowest prior).
        """
        if not states:
            return {}

        n = len(states)
        if n == 1:
            return {states[0].problem_id: 1.0}

        # Sort ascending by Q  →  index 0 = lowest Q
        sorted_states = sorted(states, key=lambda s: s.Q)

        # Assign ranks: lowest Q gets highest rank (= n), highest Q gets rank 1
        rank_sum = n * (n + 1) / 2  # sum of 1..n
        priors: dict[str, float] = {}
        for idx, state in enumerate(sorted_states):
            rank = n - idx          # first in sorted list (lowest Q) → rank n
            priors[state.problem_id] = rank / rank_sum

        return priors

    def select_state(self, archive: Archive) -> str:
        """Select the state with the highest PUCT score.

        Returns the problem_id of the selected state.

        If all states have 0 visits (T == 0), fall back to uniform random
        selection to bootstrap the search.
        """
        all_states = archive.get_all_states()
        if not all_states:
            raise ValueError("Archive is empty — cannot select a state.")

        # Fallback: if nobody has been visited yet, pick at random
        if archive.total_visits == 0:
            chosen = random.choice(all_states)
            logger.info(
                "T=0 → random selection: problem '%s'",
                chosen.problem_id,
            )
            return chosen.problem_id

        # Compute PUCT scores and pick the argmax
        scores = self.compute_scores(archive)
        best_pid = max(scores, key=scores.get)  # type: ignore[arg-type]

        logger.info(
            "PUCT selected problem '%s' (score=%.4f, Q=%.4f, n=%d, T=%d)",
            best_pid,
            scores[best_pid],
            archive.get_state(best_pid).Q,  # type: ignore[union-attr]
            archive.get_state(best_pid).visit_count,  # type: ignore[union-attr]
            archive.total_visits,
        )
        if logger.isEnabledFor(logging.DEBUG):
            for pid, sc in sorted(scores.items(), key=lambda kv: -kv[1]):
                s = archive.get_state(pid)
                assert s is not None
                logger.debug(
                    "  %-30s  score=%.4f  Q=%.4f  n=%d",
                    pid,
                    sc,
                    s.Q,
                    s.visit_count,
                )

        return best_pid

    def compute_scores(self, archive: Archive) -> dict[str, float]:
        """Compute PUCT scores for all states. Useful for logging."""
        all_states = archive.get_all_states()
        if not all_states:
            return {}

        priors = self.compute_prior(all_states)
        raw_scale = archive.get_reward_scale()
        scale = max(raw_scale, self.SCALE_FLOOR)
        T = archive.total_visits

        scores: dict[str, float] = {}
        for state in all_states:
            exploration = (
                self.c
                * scale
                * priors[state.problem_id]
                * math.sqrt(1 + T)
                / (1 + state.visit_count)
            )
            scores[state.problem_id] = state.Q + exploration

        return scores
