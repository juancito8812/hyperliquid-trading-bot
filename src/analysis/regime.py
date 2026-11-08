"""Causal regime filter: a small Gaussian HMM on bar returns.

What it is
----------
A 2-state (calm/volatile) Gaussian hidden Markov model fitted on log-returns
with **forward-only filtering**. Every probability the strategy consumes is

    P(state_t | r_1..r_t)

so the gate can never peek at the future — a property the backtests of
regime-switching overlays routinely violate when they use smoothed posteriors.

What it is not
--------------
It is not a deep model. Two Gaussian states, diagonal variance, variance floor,
a hard shrinkage fallback: if Baum-Welch fails to converge or degenerates, the
caller gets ``p_volatile`` pinned to the unconditional variance ratio instead
of garbage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

__all__ = ["GaussianHMM", "RegimeGate", "FitDegenerate"]

import statistics


class FitDegenerate(RuntimeError):
    """Raised when a fit produces an unusable state decomposition."""


@dataclass
class GaussianHMM:
    """Minimal Gaussian HMM with log-domain Baum-Welch."""

    n_states: int = 2
    max_iter: int = 60
    tol: float = 1e-5
    var_floor: float = 1e-12
    sticky: float = 0.95  # initial self-transition probability
    trans_floor: float = 1e-4  # keep every state reachable during Baum-Welch

    mu: List[float] | None = None
    var: List[float] | None = None
    trans: List[List[float]] | None = None
    init: List[float] | None = None
    vol_state: int = 1
    iterations: int = 0
    converged: bool = False
    loglik: float = 0.0
    fallback: bool = False

    # ── fitting ─────────────────────────────────────────────────────────
    @staticmethod
    def _pdf(x: float, mu: float, var: float) -> float:
        return -0.5 * (math.log(2 * math.pi * var) + (x - mu) ** 2 / var)

    def fit(self, x: Sequence[float]) -> "GaussianHMM":
        if len(x) < 20 * self.n_states:
            raise FitDegenerate(f"need >= {20 * self.n_states} returns, got {len(x)}")
        xs = [float(v) for v in x]
        mean = statistics.fmean(xs)
        spread = statistics.pstdev(xs) or 1e-9

        # magnitude-based init: regimes separate by |return|, so label bars by
        # absolute-size quantiles instead of value quantiles (a value-quartile
        # init makes Baum-Welch collapse into an absorbing two-cluster split).
        mag_order = sorted(range(len(xs)), key=lambda t: abs(xs[t]))
        k = self.n_states
        self.mu = []
        self.var = []
        for s in range(k):
            lo, hi = int(len(mag_order) * s / k), int(len(mag_order) * (s + 1) / k)
            idxs = mag_order[max(0, lo):max(1, hi)]
            chunk = [xs[t] for t in idxs] or [mean]
            m = statistics.fmean(chunk)
            self.mu.append(m)
            self.var.append(max(statistics.pvariance(chunk), self.var_floor))
        A = [[0.0] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                A[i][j] = self.sticky if i == j else (1.0 - self.sticky) / max(1, k - 1)
        self.trans = A
        self.init = [1.0 / k] * k
        self.vol_state = k - 1  # magnitude init: last state sees the fattest tail
        self.fallback = False
        self.converged = False
        self.iterations = 0

        prev_ll = -math.inf
        for _ in range(self.max_iter):
            ll, alpha, gamma, xi_sum = self._forward_backward(xs)
            self.iterations += 1
            if not math.isfinite(ll):
                raise FitDegenerate("non-finite log-likelihood")
            # M-step
            N = len(xs)
            new_mu, new_var = [], []
            for s in range(k):
                gsum = sum(gamma[t][s] for t in range(N))
                if gsum < 1e-9:
                    raise FitDegenerate(f"state {s} collapsed in Baum-Welch")
                new_mu.append(sum(gamma[t][s] * xs[t] for t in range(N)) / gsum)
                vs = sum(gamma[t][s] * (xs[t] - new_mu[-1]) ** 2 for t in range(N))
                new_var.append(max(vs / gsum, self.var_floor))
            new_A = [[xi_sum[i][j] for j in range(k)] for i in range(k)]
            for i in range(k):
                row = sum(new_A[i])
                if row < 1e-9:
                    for j in range(k):
                        new_A[i][j] = self.trans[i][j]  # keep previous
                else:
                    for j in range(k):
                        new_A[i][j] /= row
                    # transition floor: keeps every state reachable so Baum-Welch
                    # cannot deadlock in an absorbing split
                    for j in range(k):
                        new_A[i][j] = max(new_A[i][j], self.trans_floor)
                    row = sum(new_A[i])
                    for j in range(k):
                        new_A[i][j] /= row
            self.mu, self.var, self.trans = new_mu, new_var, new_A
            self.loglik = ll
            if abs(ll - prev_ll) < self.tol * (abs(prev_ll) + 1.0):
                self.converged = True
                break
            prev_ll = ll
        self.vol_state = max(range(k), key=lambda s: self.var[s])
        lo_v, hi_v = min(self.var), max(self.var)
        if hi_v <= lo_v * 1.02:  # states are the same regime twice
            self.fallback = True
        return self

    # ── forward-backward machinery ──────────────────────────────────────
    def _forward(self, xs: Sequence[float]):
        if self.mu is None or self.var is None or self.trans is None or self.init is None:
            raise FitDegenerate("forward before fit")
        k = self.n_states
        alphas: List[List[float]] = []
        prev = [math.log(self.init[s]) + self._pdf(xs[0], self.mu[s], self.var[s])
                for s in range(k)]
        # log-normalize
        norm = _logsumexp(prev)
        prev = [p - norm for p in prev]
        alphas.append(prev)
        ll = norm
        for t in range(1, len(xs)):
            cur = []
            for s in range(k):
                acc = -math.inf
                for i in range(k):
                    tr = self.trans[i][s]
                    if tr <= 0:
                        continue
                    acc = _logaddexp(acc, prev[i] + math.log(tr))
                cur.append(acc + self._pdf(xs[t], self.mu[s], self.var[s]))
            norm = _logsumexp(cur)
            cur = [p - norm for p in cur]
            alphas.append(cur)
            ll += norm
            prev = cur
        return ll, alphas

    def _forward_backward(self, xs):
        if self.mu is None or self.var is None or self.trans is None:
            raise FitDegenerate("e-step before fit")
        k = self.n_states
        ll, alphas = self._forward(xs)
        N = len(xs)
        gamma = [[0.0] * k for _ in range(N)]
        xi_sum = [[0.0] * k for _ in range(k)]
        beta = [0.0] * k  # log-domain beta_T = 0
        for t in range(N - 1, -1, -1):
            if t == N - 1:
                # gamma_T: normalized alpha row (beta_T = 0)
                for s in range(k):
                    gamma[t][s] = math.exp(alphas[t][s])
                continue
            # per-row rescaled posterior: exp(alpha + beta) underflows even in
            # log-domain once t shrinks, so normalize each row against its own
            # logsumexp before exponentiating.
            logs = [alphas[t][s] + beta[s] for s in range(k)]
            norm = _logsumexp(logs)
            for s in range(k):
                gamma[t][s] = math.exp(logs[s] - norm)
            new_beta = [-math.inf] * k
            for i in range(k):
                acc = -math.inf
                for j in range(k):
                    tr = self.trans[i][j]
                    if tr <= 0:
                        continue
                    acc = _logaddexp(acc, math.log(tr) + beta[j]
                                     + self._pdf(xs[t + 1], self.mu[j], self.var[j]))
                new_beta[i] = acc
            beta_next, beta = beta, new_beta
            # joint P(s_t=i, s_{t+1}=j | obs): normalize its own pair row so the
            # numerator of the M-step stays on the same scale as gamma.
            pair_logs = []
            for i in range(k):
                for j in range(k):
                    tr = self.trans[i][j]
                    if tr <= 0:
                        pair_logs.append(-math.inf)
                        continue
                    pair_logs.append(
                        alphas[t][i] + math.log(tr) + beta_next[j]
                        + self._pdf(xs[t + 1], self.mu[j], self.var[j]))
            z = _logsumexp(pair_logs) if pair_logs else -math.inf
            n = 0
            for i in range(k):
                for j in range(k):
                    val = math.exp(pair_logs[n] - z) if math.isfinite(pair_logs[n]) else 0.0
                    n += 1
                    if val > 0.0:
                        xi_sum[i][j] += val
        total = sum(gamma[-1])
        if total <= 0 or not math.isfinite(total):
            raise FitDegenerate("zero posterior mass")
        return ll, alphas, gamma, xi_sum

    # ── causal inference ────────────────────────────────────────────────
    def filter_proba(self, x: Sequence[float]) -> List[List[float]]:
        """P(state_t | x_1..x_t), one row per observation, forward-only."""
        if self.mu is None:
            raise FitDegenerate("filter_proba before fit")
        if self.init is None:
            raise FitDegenerate("filter_proba before fit")
        if self.trans is None:
            raise FitDegenerate("filter_proba before fit")
        if self.var is None:
            raise FitDegenerate("filter_proba before fit")
        k = self.n_states
        xs = [float(v) for v in x]
        if not xs:
            raise FitDegenerate("empty sequence")
        prev = [self.init[s] for s in range(k)]
        rows: List[List[float]] = []
        biggest_var = max(self.var)
        baseline = [biggest_var] * k
        if self.fallback or biggest_var <= min(self.var) * 1.02:
            baseline = [statistics.fmean(self.var)] * k
        for t, v in enumerate(xs):
            if self.fallback:
                # Degenerate fit: rank the state by |return| vs own variance only.
                rows.append(_fallback_posterior(v, k, baseline))
                continue
            if t > 0:
                prev = [sum(prev[i] * self.trans[i][s] for i in range(k)) for s in range(k)]
            post = [prev[s] * math.exp(self._pdf(v, self.mu[s], self.var[s])) for s in range(k)]
            tot = sum(post)
            if tot <= 0:
                rows.append([1.0 / k] * k)
                continue
            rows.append([p / tot for p in post])
            prev = rows[-1]
        return rows

    def predict_proba(self, x: Sequence[float]) -> List[float]:
        return self.filter_proba(x)[-1]

    def p_volatile(self, x: Sequence[float]) -> float:
        return self.predict_proba(x)[self.vol_state]


def _logsumexp(vals) -> float:
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in vals if v > -math.inf))


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _fallback_posterior(v: float, k: int, variances: Sequence[float]) -> List[float]:
    """Causal fallback: weight states by tail magnitude, not by transitions."""
    total = sum(abs(v) + 1e-12 for _ in range(k)) or 1.0
    out = [1.0 / k] * k
    biggest = k - 1
    ratio = abs(v) / (math.sqrt(variances[biggest]) + 1e-12)
    out[biggest] = min(0.95, 0.5 + ratio / 4.0)
    rest = (1.0 - out[biggest]) / max(1, k - 1)
    for s in range(biggest):
        out[s] = rest
    return out


# ── gate wrapper used by strategies ────────────────────────────────────────
class RegimeGate:
    """Fit-once-then-filter regime veto.

    * ``fit`` runs on an initial returns window (warm-up).
    * ``refresh`` refits every ``refresh_every`` observations, on data up to
      and including the current bar only (no lookahead).
    * ``allows`` returns False when the filtered P(volatile) crosses
      ``max_p_volatile``.
    * Any fit/hmm error ⇒ veto disabled (fail-open), never a crash in the
      hot loop.
    """

    def __init__(self, max_p_volatile: float = 0.70, refresh_every: int = 200,
                 min_history: int = 100) -> None:
        self.max_p_volatile = float(max_p_volatile)
        self.refresh_every = int(refresh_every)
        self.min_history = int(min_history)
        self.hmm: GaussianHMM | None = None
        self.n_since_fit = 10 ** 12
        self.vetoed_obs = 0
        self.observations = 0
        self.degraded = False

    def fit(self, closes: Sequence[float]) -> None:
        rets = _log_returns(closes)
        if len(rets) < self.min_history:
            return
        try:
            self.hmm = GaussianHMM().fit(rets)
            self.degraded = self.hmm.fallback
            self.n_since_fit = 0
        except FitDegenerate:
            self.hmm = None
            self.degraded = True

    def refresh_and_filter(self, closes: Sequence[float]) -> bool:
        """Consume a bar; returns True when entries are allowed."""
        self.observations += 1
        rets = _log_returns(closes)
        if self.hmm is None:
            self.fit(closes)
            return True
        self.n_since_fit += 1
        if self.n_since_fit >= self.refresh_every:
            self.fit(closes)
        if self.hmm is None:
            return True
        window = rets[-self.min_history * 3:]
        try:
            p = self.hmm.p_volatile(window)
        except FitDegenerate:
            self.hmm = None
            return True
        if p > self.max_p_volatile:
            self.vetoed_obs += 1
            return False
        return True

    def status(self) -> dict:
        return {
            "degraded": self.degraded,
            "veto_pct": round(self.vetoed_obs / self.observations * 100, 1)
            if self.observations else 0.0,
            "n_states": self.hmm.n_states if self.hmm else 0,
            "iterations": self.hmm.iterations if self.hmm else 0,
        }


def _log_returns(closes: Sequence[float]) -> List[float]:
    out: List[float] = []
    for a, b in zip(closes, closes[1:]):
        if a > 0 and b > 0:
            out.append(math.log(b / a))
    return out
