"""CreditRisk+ against closed forms and a brute-force simulation of the same model."""

import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from eqlx_portfolio import InputError, creditriskplus, default_correlation, sigma_from_asset_correlation


def test_one_sector_equal_loans_is_negative_binomial():
    n, p, sigma = 50, 0.02, 0.8
    res = creditriskplus(np.ones(n), np.full(n, p), np.ones((n, 1)), [sigma], loss_unit=1.0)
    mu, alpha = n * p, 1 / sigma**2
    ref = nbinom.pmf(np.arange(res.probabilities.size), alpha, alpha / (alpha + mu))
    assert res.probabilities == pytest.approx(ref, abs=1e-14)
    assert res.expected_loss == pytest.approx(mu)
    assert res.sd == pytest.approx(np.sqrt(mu + (mu * sigma) ** 2))


def test_without_sector_volatility_it_is_poisson():
    n, p = 50, 0.02
    for weights, sigma in ((np.zeros((n, 1)), [0.8]), (np.ones((n, 1)), [0.0])):
        res = creditriskplus(np.ones(n), np.full(n, p), weights, sigma, loss_unit=1.0)
        assert res.probabilities == pytest.approx(poisson.pmf(np.arange(res.probabilities.size), n * p), abs=1e-14)


def test_two_loans_by_hand():
    # Loans of 1 and 2 units, no common factor: P(loss = 0) = exp(-(p1 + p2)), P(loss = 1) = p1 * that, ...
    res = creditriskplus([1.0, 2.0], [0.1, 0.2], np.zeros((2, 1)), [1.0], loss_unit=1.0)
    p0 = np.exp(-0.3)
    assert res.probabilities[:4] == pytest.approx(
        [p0, 0.1 * p0, (0.1**2 / 2 + 0.2) * p0, (0.1**3 / 6 + 0.1 * 0.2) * p0]
    )


@pytest.fixture(scope="module")
def mixed():
    rng = np.random.default_rng(5)
    n = 60
    pd = rng.uniform(0.005, 0.08, n)
    units = rng.integers(1, 30, n).astype(float)
    weights = rng.dirichlet([1, 1, 1], n) * 0.8
    sigma = np.array([0.5, 1.0, 1.5])
    res = creditriskplus(units, pd, weights, sigma, loss_unit=1.0, confidence=[0.95, 0.99])
    scenarios = 1_000_000
    gamma = np.column_stack([rng.gamma(1 / s**2, s**2, scenarios) for s in sigma])
    counts = rng.poisson(pd[None, :] * ((1 - weights.sum(axis=1))[None, :] + gamma @ weights.T))
    return res, counts, units


def test_distribution_matches_a_simulation_of_the_same_model(mixed):
    res, counts, units = mixed
    loss = counts @ units
    assert res.expected_loss == pytest.approx(loss.mean(), rel=0.005)
    assert res.sd == pytest.approx(loss.std(), rel=0.01)
    assert res.tail_mass < 1e-9
    empirical = np.bincount(loss.astype(int), minlength=400)[:400] / loss.size
    assert np.abs(np.cumsum(empirical) - res.cdf()[:400]).max() < 0.002
    for tail in res.tails:
        assert tail.value_at_risk == pytest.approx(np.quantile(loss, tail.confidence, method="inverted_cdf"), abs=2)


def test_contributions_match_the_simulation_and_add_up(mixed):
    res, counts, units = mixed
    loss = counts @ units
    for tail in res.tails:
        worst = loss >= tail.value_at_risk
        assert tail.expected_shortfall == pytest.approx(loss[worst].mean(), rel=0.01)
        assert tail.contributions.sum() == pytest.approx(tail.expected_shortfall, rel=1e-12)
        simulated = (counts[worst] * units).mean(axis=0)
        assert tail.contributions == pytest.approx(simulated, abs=0.06 * tail.contributions.max() + 0.05)


def test_rounding_to_loss_units_keeps_each_expected_loss():
    amount = np.array([1234.0, 5678.0, 91011.0])
    pd = np.array([0.01, 0.02, 0.03])
    res = creditriskplus(amount, pd, np.ones((3, 1)), [0.7], loss_unit=1000.0)
    assert res.units == pytest.approx([1, 6, 91])
    assert res.units * res.loss_unit * res.adjusted_pd == pytest.approx(amount * pd)
    assert res.expected_loss == pytest.approx(float(amount @ pd))
    fine = creditriskplus(amount, pd, np.ones((3, 1)), [0.7], loss_unit=100.0)
    assert res.sd == pytest.approx(fine.sd, rel=0.01)


def test_default_loss_unit_keeps_a_fine_grained_portfolio_fast():
    n = 100_000
    res = creditriskplus(np.full(n, 1000.0), np.full(n, 0.02), np.ones((n, 1)), [0.5], confidence=[0.999])
    assert res.probabilities.size <= 45_000 and res.tail_mass < 1e-9
    assert res.expected_loss == pytest.approx(n * 1000.0 * 0.02)
    # one sector, equal loans: the number of defaults is negative binomial whatever the unit
    mu, alpha = n * 0.02, 4.0
    exact_var = 1000.0 * nbinom.ppf(0.999, alpha, alpha / (alpha + mu))
    assert res.tails[0].value_at_risk == pytest.approx(exact_var, rel=0.002)


def test_large_portfolio_stays_accurate():
    # 20,000 borrowers, about 550 expected defaults: P(no loss) is far below the smallest double.
    rng = np.random.default_rng(1)
    n = 20_000
    pd = rng.uniform(0.005, 0.05, n)
    amount = rng.lognormal(12, 1, n)
    weights = np.zeros((n, 5))
    weights[np.arange(n), rng.integers(0, 5, n)] = 1.0
    res = creditriskplus(amount, pd, weights, [0.6] * 5, confidence=[0.999])
    assert np.all(np.isfinite(res.probabilities)) and np.all(res.probabilities >= 0)
    assert res.probabilities.sum() == pytest.approx(1.0, abs=1e-9)
    grid = np.arange(res.probabilities.size) * res.loss_unit
    mean = float(grid @ res.probabilities)
    assert mean == pytest.approx(res.expected_loss, rel=1e-9)
    assert np.sqrt(float((grid - mean) ** 2 @ res.probabilities)) == pytest.approx(res.sd, rel=1e-6)
    tail = res.tails[0]
    assert tail.contributions.sum() == pytest.approx(tail.expected_shortfall, rel=1e-9)
    assert tail.expected_shortfall > tail.value_at_risk > res.expected_loss


def test_sector_volatility_from_asset_correlation_gives_the_same_default_correlation():
    pd, rho = 0.02, 0.2
    sigma = sigma_from_asset_correlation(pd, rho)
    # In CreditRisk+ two borrowers fully in one sector have default covariance pd^2 sigma^2.
    implied = (pd * sigma) ** 2 / (pd * (1 - pd))
    assert implied == pytest.approx(float(default_correlation(pd, pd, rho)[0]))
    assert sigma_from_asset_correlation(pd, 0.0) == pytest.approx(0.0, abs=1e-7)


def test_bad_input_is_refused():
    ok = dict(amount=[1.0, 2.0], pd=[0.1, 0.2], sector_weights=np.ones((2, 1)), sector_sigma=[0.5])
    for change in (
        {"amount": [1.0, 0.0]},
        {"pd": [0.1, 1.2]},
        {"sector_weights": np.full((2, 2), 0.6), "sector_sigma": [0.5, 0.5]},
        {"sector_sigma": [0.5, 0.5]},
        {"sector_sigma": [-0.1]},
        {"loss_unit": 0.0},
        {"loss_unit": 100.0},
        {"confidence": [1.0]},
    ):
        with pytest.raises(InputError):
            creditriskplus(**{**ok, **change})
