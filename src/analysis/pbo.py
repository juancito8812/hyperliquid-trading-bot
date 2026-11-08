"""Probability of Backtest Overfitting (PBO) via CSCV.

Implements Bailey, Borwein, López de Prado & Zhu (2014) *Pseudo-Mathematics
and Financial Gambler's Tests* combinatorially-split cross-validation:

1. Take ``N`` candidate configurations and their aligned per-bar returns.
2. Split the timeline into ``S`` contiguous blocks.
3. For every combination of ``S/2`` blocks as **in-sample** (IS) and the rest
   as **out-of-sample** (OOS):
   * pick the config with the best IS Sharpe;
   * record where that config lands on the OOS ranking (``rank_pct``,
     0 = best, 1 = worst).
4. ``PBO`` = share of trials whose OOS rank fell in the worst half
   (``logit < 0``). PBO ≈ 0.5 = the selection is a coin flip; PBO > 0.5 =
   picking the IS winner loses money OOS on average.

CSCV is finite-set here: no shuffled indices, blocks stay contiguous so
serial dependence is preserved. With ``max_combos`` set, trials are sampled
deterministically with a fixed seed instead of enumerating all C(S, S/2).
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

__all__ = ["CscvResult", "cscv_pbo"]


@dataclass
class CscvResult:
    pbo: float
    n_combos: int
    n_configs: int
    n_blocks: int
    logit_mean: float
    logit_q25: float
    logit_q75: float
    mean_oos_rank_pct: float
    lambdas: List[float]
    is_picked: List[int]  # per trial, index of the IS-best config


def _sharpe(returns: Sequence[float]) -> float:
    if not returns:
        return -math.inf
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = math.sqrt(var)
    if std <= 1e-12:
        return 0.0 if mean > 0 else -0.0
    return mean / std


def cscv_pbo(
    returns_matrix: Sequence[Sequence[float]],
    n_blocks: int = 10,
    max_combos: int = 20000,
    seed: int = 7,
) -> CscvResult:
    """``returns_matrix`` = aligned per-block returns, shape (T, N)."""
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be even and >= 2")
    T = len(returns_matrix)
    N = len(returns_matrix[0]) if T else 0
    if N < 2:
        raise ValueError("need at least 2 candidate configurations")
    if T < n_blocks:
        raise ValueError(f"need >= {n_blocks} aligned periods, got {T}")
    for row in returns_matrix:
        if len(row) != N:
            raise ValueError("ragged returns matrix")

    rows = [[float(v) for v in r] for r in returns_matrix]
    block = T // n_blocks
    blocks: List[List[List[float]]] = []
    for s in range(n_blocks):
        start = s * block
        end = start + block if s < n_blocks - 1 else T
        blocks.append(rows[start:end])

    all_combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if len(all_combos) > max_combos:
        rng = random.Random(seed)
        all_combos = rng.sample(all_combos, max_combos)

    lambdas: List[float] = []
    is_picked: List[int] = []
    ranks_pct: List[float] = []

    for combo in all_combos:
        is_idx = set(combo)
        oos_idx = [b for b in range(n_blocks) if b not in is_idx]

        is_matrix = [r for b in sorted(is_idx) for r in blocks[b]]
        oos_matrix = [r for b in oos_idx for r in blocks[b]]

        is_sharpes = [_sharpe(col) for col in zip(*is_matrix)]
        oos_sharpes = [_sharpe(col) for col in zip(*oos_matrix)]

        best = max(range(N), key=lambda i: is_sharpes[i])
        is_picked.append(best)

        oos_sorted = sorted(range(N), key=lambda i: oos_sharpes[i])
        rank = oos_sorted.index(best)  # ascending: 0 = worst OOS perf
        omega = (rank + 1) / (N + 1)
        # standard CSCV logit: positive = IS winner stays above median OOS
        lambdas.append(math.log(omega / (1.0 - omega)))
        ranks_pct.append(rank / max(1, N - 1))

    pbo = sum(1 for l in lambdas if l <= 0.0) / len(lambdas)

    sorts = sorted(lambdas)
    def q(p: float) -> float:
        i = min(len(sorts) - 1, max(0, int(p * (len(sorts) - 1))))
        return sorts[i]

    return CscvResult(
        pbo=round(pbo, 4),
        n_combos=len(lambdas),
        n_configs=N,
        n_blocks=n_blocks,
        logit_mean=round(sum(lambdas) / len(lambdas), 4),
        logit_q25=round(q(0.25), 4),
        logit_q75=round(q(0.75), 4),
        mean_oos_rank_pct=round(sum(ranks_pct) / len(ranks_pct), 4),
        lambdas=[round(l, 4) for l in lambdas],
        is_picked=is_picked,
    )
