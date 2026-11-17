"""Pure score calculations for the Cycle A training-membership protocol."""

from __future__ import annotations

import math
from collections.abc import Sequence


def summarize_token_nll(token_nll: Sequence[float], k_percent: float = 20.0) -> tuple[float, float, int]:
    """Return mean NLL and mean NLL over the k% least-likely tokens.

    Least-likely tokens have the largest negative log likelihood. A member
    signal is formed later by negating these losses, so higher means more
    member-like.
    """
    if not token_nll:
        raise ValueError("token_nll must not be empty")
    if not 0.0 < k_percent <= 100.0:
        raise ValueError("k_percent must be in (0, 100]")
    losses = [float(value) for value in token_nll]
    if not all(math.isfinite(value) and value >= 0.0 for value in losses):
        raise ValueError("token losses must be finite and non-negative")
    selected_count = max(1, math.ceil(len(losses) * k_percent / 100.0))
    least_likely_losses = sorted(losses, reverse=True)[:selected_count]
    return sum(losses) / len(losses), sum(least_likely_losses) / selected_count, len(losses)
