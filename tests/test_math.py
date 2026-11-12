"""Building blocks against independent truth: SciPy, closed forms, limits."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from eqlx_portfolio import (
    InputError,
    basel_corporate_correlation,
    conditional_pd,
    default_correlation,
    joint_default_probability,
    joint_migration,
    large_portfolio_loss,
    migration_thresholds,
    percentile_of_states,
    sector_factors,
    single_factor,
    value_stats,
)
from eqlx_portfolio._bvn import bvn_cdf
from eqlx_portfolio.vasicek import loss_rate_cdf


@pytest.mark.parametrize("rho", [-0.999, -0.9, -0.3, 0.0, 0.2, 0.5, 0.9, 0.99, 0.9999])
def test_bivariate_normal_matches_scipy(rho):
    rng = np.random.default_rng(int(abs(rho) * 1e4))
    pts = rng.normal(scale=2.0, size=(60, 2))
    ref = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf(pts)
    assert bvn_cdf(pts[:, 0], pts[:, 1], rho) == pytest.approx(ref, abs=1e-12)


def test_bivariate_normal_limits():
    h, k = -1.3, 0.4
    assert float(bvn_cdf(h, k, 1.0)) == pytest.approx(norm.cdf(min(h, k)))
    assert float(bvn_cdf(h, k, -1.0)) == pytest.approx(max(0.0, norm.cdf(h) + norm.cdf(k) - 1))
    assert float(bvn_cdf(h, k, 0.0)) == pytest.approx(norm.cdf(h) * norm.cdf(k))
    assert float(bvn_cdf(np.inf, k, 0.4)) == pytest.approx(norm.cdf(k))
    assert float(bvn_cdf(-np.inf, k, 0.4)) == 0.0
    assert float(bvn_cdf(h, k, 0.3)) == pytest.approx(float(bvn_cdf(k, h, 0.3)))
    # deep tail: 0.03% twice, a typical investment-grade pair
    z = norm.ppf(0.0003)
    ref = multivariate_normal(mean=[0, 0], cov=[[1, 0.2], [0.2, 1]]).cdf([z, z])
    assert float(bvn_cdf(z, z, 0.2)) == pytest.approx(ref, rel=1e-9)


def test_joint_default_probability_properties():
    assert joint_default_probability(0.02, 0.05, 0.0)[0] == pytest.approx(0.001)
    assert joint_default_probability(0.02, 0.05, 1.0)[0] == pytest.approx(0.02)
    grid = joint_default_probability(0.02, 0.05, np.linspace(0, 0.9, 10))
    assert np.all(np.diff(grid) > 0)
    assert default_correlation(0.02, 0.02, 1.0)[0] == pytest.approx(1.0)
    assert default_correlation(0.02, 0.05, 0.0)[0] == pytest.approx(0.0, abs=1e-12)
    assert 0 < default_correlation(0.01, 0.01, 0.2)[0] < 0.2  # default correlation is far below asset correlation
    with pytest.raises(InputError):
        default_correlation(0.0, 0.05, 0.2)
    with pytest.raises(InputError):
        joint_default_probability(0.02, 0.05, 1.2)


def test_joint_migration_is_a_distribution_with_the_right_margins():
    a = [0.01, 0.04, 0.90, 0.03, 0.02]
    b = [0.00, 0.10, 0.80, 0.00, 0.10]  # states with zero probability
    for rho in (-0.5, 0.0, 0.45, 1.0):
        j = joint_migration(a, b, rho)
        assert j.sum() == pytest.approx(1.0)
        assert j.sum(axis=1) == pytest.approx(a, abs=1e-12) and j.sum(axis=0) == pytest.approx(b, abs=1e-12)
        assert np.all(j >= 0)
    assert joint_migration(a, a, 1.0) == pytest.approx(np.diag(a), abs=1e-12)
    # worst-with-worst gets likelier with correlation
    assert joint_migration(a, b, 0.45)[-1, -1] > joint_migration(a, b, 0.0)[-1, -1]


def test_thresholds_recover_the_probabilities():
    p = np.array([0.002, 0.05, 0.90, 0.04, 0.008])
    z = migration_thresholds(p)
    assert norm.cdf(z[-1]) == pytest.approx(p[-1])
    assert 1 - norm.cdf(z[0]) == pytest.approx(p[0])
    assert np.all(np.diff(z) < 0)
    with pytest.raises(InputError):
        migration_thresholds([0.5, 0.6])
    with pytest.raises(InputError):
        migration_thresholds([1.0])


def test_value_stats_and_percentile():
    s = value_stats([0.9, 0.1], [100.0, 40.0])
    assert s.mean == pytest.approx(94.0) and s.variance == pytest.approx(0.9 * 36 + 0.1 * 54**2)
    assert value_stats([0.9, 0.1], [100.0, 40.0], [0.0, 20.0]).variance == pytest.approx(s.variance + 0.1 * 400)
    assert percentile_of_states([0.9, 0.1], [100.0, 40.0], 0.05) == 40.0
    assert percentile_of_states([0.9, 0.1], [100.0, 40.0], 0.10) == 40.0
    assert percentile_of_states([0.9, 0.1], [100.0, 40.0], 0.11) == 100.0
    with pytest.raises(InputError):
        value_stats([0.9, 0.1], [100.0])


def test_large_portfolio_formula_matches_the_basel_risk_weight_function():
    # Basel II, corporate exposures, PD 1%, LGD 45%, one-year maturity (no maturity adjustment):
    # K = LGD * [N((G(PD) + sqrt(R) G(0.999)) / sqrt(1 - R)) - PD] with R from the correlation formula.
    pd, lgd = 0.01, 0.45
    r = float(basel_corporate_correlation([pd])[0])
    assert r == pytest.approx(
        0.12 * (1 - np.exp(-0.5)) / (1 - np.exp(-50)) + 0.24 * (1 - (1 - np.exp(-0.5)) / (1 - np.exp(-50)))
    )
    k = lgd * (norm.cdf((norm.ppf(pd) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)) - pd)
    assert large_portfolio_loss([lgd], [pd], r, 0.999) - lgd * pd == pytest.approx(k)
    assert basel_corporate_correlation([0.0, 1.0]) == pytest.approx([0.24, 0.12])
    assert conditional_pd([0.02], 0.0, 0.999)[0] == pytest.approx(0.02)
    # Vasicek's loss-rate distribution inverts the conditional default rate
    q = conditional_pd([0.02], 0.2, 0.95)[0]
    assert loss_rate_cdf([q], 0.02, 0.2)[0] == pytest.approx(0.95)


def test_factor_models_give_the_stated_pair_correlations():
    m = single_factor([0.25, 0.16, 0.09], 3)
    assert m.correlation(0, 1) == pytest.approx(0.2) and m.correlation(1, 2) == pytest.approx(0.12)
    sector = np.array([0, 0, 1, 2])
    between = [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]]
    m = sector_factors(sector, [0.25, 0.16, 0.09], between)
    c = m.correlation_matrix()
    assert c[0, 1] == pytest.approx(0.25)
    assert c[0, 2] == pytest.approx(np.sqrt(0.25 * 0.16) * 0.5)
    assert c[2, 3] == pytest.approx(np.sqrt(0.16 * 0.09) * 0.3)
    assert m.idio == pytest.approx(np.sqrt(1 - np.array([0.25, 0.25, 0.16, 0.09])))
    with pytest.raises(InputError):
        sector_factors(sector, [0.25, 0.16, 0.09], [[1, 0.99, -0.99], [0.99, 1, 0.99], [-0.99, 0.99, 1]])
    with pytest.raises(InputError):
        single_factor(1.2, 3)


def test_factor_root_is_unique_and_handles_perfectly_correlated_sectors():
    from eqlx_portfolio.factors import _root

    between = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1.0]])
    root = _root(between, "m")
    assert root @ root.T == pytest.approx(between)
    assert np.allclose(root, np.tril(root)) and np.all(np.diag(root) > 0)  # the Cholesky factor
    singular = np.ones((2, 2))  # two sectors that are one
    r = _root(singular, "m")
    assert r @ r.T == pytest.approx(singular, abs=1e-12)
    m = sector_factors(np.array([0, 1]), [0.2, 0.2], singular)
    assert m.correlation(0, 1) == pytest.approx(0.2)
