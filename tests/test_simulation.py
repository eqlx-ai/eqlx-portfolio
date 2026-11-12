"""The simulation is reproducible and agrees with exact results where those exist."""

import numpy as np
import pytest
from scipy.stats import binom

from eqlx_portfolio import (
    InputError,
    default_thresholds,
    exact_loss_moments,
    large_portfolio_loss,
    quantiles,
    sector_factors,
    simulate,
    single_factor,
)


def _portfolio(n=120, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.005, 0.06, n), rng.lognormal(0.0, 1.0, n)


def _run(pd, amount, model, **kw):
    kw.setdefault("scenarios", 200_000)
    kw.setdefault("seed", 7)
    return simulate(model, default_thresholds(pd), np.column_stack((np.zeros(pd.size), amount)), **kw)


def test_same_seed_same_result_and_different_seed_differs():
    pd, amount = _portfolio()
    m = single_factor(0.2, pd.size)
    a, b, c = _run(pd, amount, m), _run(pd, amount, m), _run(pd, amount, m, seed=8)
    assert np.array_equal(a.totals, b.totals)
    assert not np.array_equal(a.totals, c.totals)


def test_mean_and_standard_deviation_match_the_exact_pairwise_formulas():
    pd, amount = _portfolio()
    sector = np.arange(pd.size) % 3
    m = sector_factors(sector, [0.25, 0.15, 0.10], [[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    sim = _run(pd, amount, m, scenarios=400_000)
    mean, sd = exact_loss_moments(amount, pd, m)
    assert mean == pytest.approx(float(amount @ pd))
    assert sim.totals.mean() == pytest.approx(mean, abs=4 * sd / np.sqrt(400_000))
    assert sim.totals.std() == pytest.approx(sd, rel=0.02)


def test_exact_standard_deviation_of_two_loans_by_hand():
    # Two loans of 1, pd 10%, independent: variance 2 * 0.09; perfectly correlated: (2^2) * 0.09.
    pd = np.array([0.1, 0.1])
    assert exact_loss_moments([1, 1], pd, single_factor(0.0, 2))[1] == pytest.approx(np.sqrt(0.18))
    assert exact_loss_moments([1, 1], pd, single_factor(1.0, 2))[1] == pytest.approx(np.sqrt(0.36))
    assert exact_loss_moments([1, 1], pd, single_factor(0.0, 2), amount_sd=[0.5, 0.5])[1] == pytest.approx(
        np.sqrt(0.18 + 2 * 0.1 * 0.25)
    )


def test_independent_equal_loans_follow_the_binomial_distribution():
    n, p = 40, 0.05
    sim = _run(np.full(n, p), np.ones(n), single_factor(0.0, n), scenarios=400_000)
    counts = np.bincount(sim.totals.astype(int), minlength=n + 1) / sim.scenarios
    assert counts[:8] == pytest.approx(binom.pmf(np.arange(8), n, p), abs=0.004)


def test_contributions_add_up_to_expected_shortfall():
    pd, amount = _portfolio()
    sim = _run(pd, amount, single_factor(0.2, pd.size), tail_levels=[0.95, 0.99, 0.999])
    ordered = np.sort(sim.totals)
    for tail in sim.tails:
        k = int(np.ceil(tail.confidence * sim.scenarios)) - 1
        assert tail.value_at_risk == ordered[k]
        assert tail.expected_shortfall == pytest.approx(ordered[k:].mean())
        assert tail.contributions.sum() == pytest.approx(tail.expected_shortfall, rel=1e-12)
        assert np.all(tail.contributions >= 0)
    # with many ties at the quantile (equal loans) the shares still add up
    n = 30
    tied = _run(np.full(n, 0.05), np.ones(n), single_factor(0.3, n), tail_levels=[0.99], scenarios=50_000)
    assert tied.tails[0].contributions.sum() == pytest.approx(tied.tails[0].expected_shortfall, rel=1e-12)
    assert tied.tails[0].contributions == pytest.approx(tied.tails[0].expected_shortfall / n, rel=0.15)


def test_a_bigger_or_riskier_loan_carries_more_risk():
    n = 50
    pd, amount = np.full(n, 0.02), np.ones(n)
    amount[0], pd[1] = 5.0, 0.10
    sim = _run(pd, amount, single_factor(0.2, n), tail_levels=[0.99], scenarios=400_000)
    c = sim.tails[0].contributions
    assert c[0] > 5 * np.median(c)  # concentration: more than five times the risk for five times the size
    assert c[1] > 2 * np.median(c)


def test_fine_portfolio_converges_to_the_large_portfolio_formula():
    n = 2000
    pd, amount = np.full(n, 0.02), np.full(n, 1.0 / n)
    sim = _run(pd, amount, single_factor(0.2, n), scenarios=400_000, seed=3)
    for q in quantiles(sim.totals, [0.99, 0.999]):
        limit = large_portfolio_loss(amount, pd, 0.2, q.confidence)
        assert q.value == pytest.approx(limit, rel=0.03)
        assert q.value >= limit * 0.99  # a finite portfolio is never meaningfully below the limit


def test_quantile_band_brackets_the_quantile():
    pd, amount = _portfolio()
    sim = _run(pd, amount, single_factor(0.2, pd.size))
    for q in quantiles(sim.totals, [0.9, 0.99, 0.999]):
        assert q.ci_low <= q.value <= q.ci_high
        assert q.expected_shortfall >= q.value
    wide = quantiles(sim.totals[:5000], [0.999])[0]
    narrow = quantiles(sim.totals, [0.999])[0]
    assert wide.ci_high - wide.ci_low > narrow.ci_high - narrow.ci_low


def test_leaving_borrowers_out_reproduces_the_run_without_them():
    pd, amount = _portfolio(60)
    base = _run(pd, amount, single_factor(0.2, 60), scenarios=50_000)
    pd2, amount2 = np.append(pd, [0.03, 0.01]), np.append(amount, [4.0, 2.0])
    flag = np.arange(62) >= 60
    both = _run(pd2, amount2, single_factor(0.2, 62), scenarios=50_000, exclude=flag, tail_levels=[0.99])
    assert both.totals_excluding is not None
    assert np.array_equal(both.totals_excluding, base.totals)
    assert np.all(both.totals >= both.totals_excluding)
    assert both.totals.mean() - base.totals.mean() == pytest.approx(4 * 0.03 + 2 * 0.01, rel=0.1)


def test_uncertain_loss_given_default_keeps_the_mean_and_adds_variance():
    n = 80
    pd, amount = np.full(n, 0.05), np.full(n, 0.5)
    m = single_factor(0.15, n)
    fixed = _run(pd, amount, m, scenarios=300_000)
    random = _run(pd, amount, m, scenarios=300_000, default_loss_rate_sd=0.2, default_loss_rate_mean=0.5)
    assert random.totals.mean() == pytest.approx(fixed.totals.mean(), rel=0.01)
    _, sd = exact_loss_moments(amount, pd, m, amount_sd=np.full(n, 0.2))
    assert random.totals.std() == pytest.approx(sd, rel=0.02)
    assert random.totals.std() > fixed.totals.std()
    with pytest.raises(InputError):
        _run(pd, amount, m, default_loss_rate_sd=0.6, default_loss_rate_mean=0.5)


def test_bad_input_is_refused():
    pd, amount = _portfolio(10)
    m = single_factor(0.2, 10)
    for kw in ({"scenarios": 10}, {"scenarios": 1000.5}, {"seed": -1}, {"tail_levels": [1.0]}):
        with pytest.raises(InputError):
            _run(pd, amount, m, **{"scenarios": 1000, **kw})
    with pytest.raises(InputError):
        _run(pd, amount, single_factor(0.2, 9))
    with pytest.raises(InputError):
        _run(pd, amount, m, exclude=np.ones(10, dtype=bool))
    with pytest.raises(InputError):
        exact_loss_moments(np.ones(1600), np.full(1600, 0.01), single_factor(0.1, 1600))
