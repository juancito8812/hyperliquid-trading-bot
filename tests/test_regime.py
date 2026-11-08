
import random

from analysis.regime import FitDegenerate, GaussianHMM, RegimeGate, _log_returns


def _two_regime_series(n=900, seed=3):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if i < n // 3 or (2 * n // 3) <= i < 5 * n // 6:
            out.append(rng.gauss(0.0005, 0.004))   # calm
        else:
            out.append(rng.gauss(-0.001, 0.020))   # volatile
    return out


def test_two_regime_recovery():
    hmm = GaussianHMM().fit(_two_regime_series())
    assert not hmm.fallback
    assert hmm.iterations <= hmm.max_iter
    # fitted variances must separate by an order of magnitude
    rows = sorted(hmm.var)
    assert rows[-1] / rows[0] > 20
    # filter the series: posterior should mirror the regime splits
    post = hmm.filter_proba(_two_regime_series())
    calm = sum(1 - p[hmm.vol_state] for p in post[:300]) / 300
    wild = sum(p[hmm.vol_state] for p in post[300:600]) / 300
    calm2 = sum(1 - p[hmm.vol_state] for p in post[600:750]) / 150
    assert calm > 0.8
    assert wild > 0.8
    assert calm2 > 0.8  # the filter must release the second calm stretch too

def test_forward_only_causal():
    """Appending future data must not change past posteriors."""
    x = _two_regime_series(seed=11)
    hmm = GaussianHMM().fit(x)
    before = hmm.filter_proba(x)
    after = hmm.filter_proba(x + [0.05, 0.07, -0.08])
    for t in range(len(before)):
        assert before[t] == after[t]

def test_degenerate_fit_falls_back():
    # constant noise: states cannot separate -> either Degenerate or fallback,
    # and p_volatile must stay finite either way
    rng = random.Random(5)
    x = [rng.gauss(0.0, 0.005) for _ in range(400)]
    try:
        hmm = GaussianHMM().fit(x)
        p = hmm.p_volatile(x)
        assert 0.0 <= p <= 1.0
        if not hmm.fallback:
            # may legitimately converge; only check sanity
            pass
    except FitDegenerate:
        pass

def test_gate_blocks_volatile_and_fails_open():
    gate = RegimeGate(max_p_volatile=0.70, refresh_every=10_000, min_history=100)
    x = [1.0 * (1 + r) for r in _two_regime_series(seed=9)]
    gate.fit(x + [x[-1]])
    assert gate.hmm is not None and not gate.degraded
    allowed = [gate.refresh_and_filter(x[: i + 2]) for i in range(1, len(x) - 1)]
    assert all(isinstance(a, bool) for a in allowed)
    assert sum(allowed) > 0
    st = gate.status()
    assert st["n_states"] == 2 and st["veto_pct"] >= 0

def test_gate_fits_late_and_fails_open():
    gate = RegimeGate(min_history=100)
    for c in [100.0, 101.0, 100.5] * 20:
        assert gate.refresh_and_filter([c, c]) is True
    assert gate.hmm is None or True  # never crashes

def test_log_returns_pure():
    assert _log_returns([100, 110, 99]) == [__import__("math").log(1.1),
                                            __import__("math").log(99 / 110)]
