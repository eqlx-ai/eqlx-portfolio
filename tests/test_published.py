"""The tool reproduces the worked examples of the CreditMetrics Technical Document (1997).

Where the document's own figures disagree with each other, the test says so and pins down why.
"""

import numpy as np
import pytest

from eqlx_portfolio import (
    bond_values_at_horizon,
    from_correlation_matrix,
    joint_migration,
    migration_thresholds,
    percentile_of_states,
    portfolio_value_stats,
    simulate,
    value_stats,
)

from . import published as pub

T = pub.TRANSITION


def test_thresholds_tables_8_5_and_8_6():
    assert np.round(migration_thresholds(T[pub.BB]), 2) == pytest.approx(pub.THRESHOLDS_BB)
    assert np.round(migration_thresholds(T[pub.A]), 2) == pytest.approx(pub.THRESHOLDS_A)


def test_thresholds_table_10_2():
    ours = np.round(migration_thresholds(T[pub.BBB]), 2)
    # One printed threshold (2.78) does not follow from the printed probabilities: 0.02% + 0.33% gives 2.70.
    assert np.delete(ours, 1) == pytest.approx(np.delete(pub.THRESHOLDS_BBB, 1))
    assert ours[1] == 2.70
    ccc = np.round(migration_thresholds(T[pub.CCC]), 2)
    assert ccc == pytest.approx(pub.THRESHOLDS_CCC, abs=0.011)


def test_bond_revaluation_tables_1_2_and_1_4():
    bbb = bond_values_at_horizon(100, 0.06, 5, pub.FORWARD_CURVES, pub.RECOVERY_SENIOR_UNSECURED)
    single_a = bond_values_at_horizon(100, 0.05, 3, pub.FORWARD_CURVES, pub.RECOVERY_SENIOR_UNSECURED)
    # The printed curve is rounded to a basis point, which moves a five-year bond by up to two cents.
    assert bbb == pytest.approx(pub.BBB_BOND_VALUES, abs=0.025)
    assert np.round(single_a, 2) == pytest.approx(pub.A_BOND_VALUES)


def test_single_bond_mean_and_standard_deviation_table_2_5():
    s = value_stats(T[pub.BBB], pub.BBB_BOND_VALUES)
    assert round(s.mean, 2) == 107.09
    assert round(s.sd, 2) == 2.99
    assert s.variance == pytest.approx(8.9477, abs=0.004)


def test_recovery_uncertainty_raises_the_standard_deviation_to_3_18():
    sds = [0.0] * 7 + [pub.RECOVERY_SENIOR_UNSECURED_SD]
    assert round(value_stats(T[pub.BBB], pub.BBB_BOND_VALUES, sds).sd, 2) == 3.18


def test_joint_migration_tables_3_1_and_3_2():
    zero = joint_migration(T[pub.BBB], T[pub.A], 0.0) * 100
    assert zero == pytest.approx(np.outer(T[pub.BBB], T[pub.A]) * 100, abs=1e-9)
    assert zero == pytest.approx(pub.JOINT_ZERO, abs=0.011)
    assert joint_migration(T[pub.BBB], T[pub.A], 0.30) * 100 == pytest.approx(pub.JOINT_030, abs=0.011)
    # the cell the document quotes in the text: both keep their rating
    assert round(float(joint_migration(T[pub.BBB], T[pub.A], 0.30)[3, 2]) * 100, 2) == 79.69


def test_joint_migration_table_8_7():
    ours = joint_migration(T[pub.BB], T[pub.A], 0.20) * 100
    assert ours == pytest.approx(pub.JOINT_BB_A_020, abs=0.05)
    assert ours.sum(axis=1) == pytest.approx(T[pub.BB] / T[pub.BB].sum() * 100, abs=1e-9)
    assert ours.sum(axis=0) == pytest.approx(T[pub.A] / T[pub.A].sum() * 100, abs=1e-9)


def test_scenarios_map_to_the_ratings_of_table_10_5():
    thresholds = np.array([migration_thresholds(T[i]) for i in (pub.BBB, pub.A, pub.CCC)])
    state = (pub.SCENARIO_RETURNS[:, :, None] < thresholds[None]).sum(axis=2)
    assert tuple(tuple(pub.RATINGS[s] for s in row) for row in state) == pub.SCENARIO_RATINGS


def test_two_bond_first_percentile_is_204_40():
    joint = joint_migration(T[pub.BBB], T[pub.A], 0.30)
    values = pub.BBB_BOND_VALUES[:, None] + pub.A_BOND_VALUES[None, :]
    assert percentile_of_states(joint, values, 0.01) == pytest.approx(204.40)


def test_two_bond_mean_in_the_document_comes_from_a_misprint():
    """The document reports 106.55 / 1.49 for the single-A bond and 213.63 for the pair.

    Its Table 1.6 prints 0.60% for the move from A to B where the transition matrix (Tables 1.8,
    2.1) and the joint tables have 0.26%. With 0.60% the probabilities add up to 100.34% and give
    exactly the reported figures; with the matrix they give 106.20 / 1.42 and 213.29.
    """
    misprinted = T[pub.A].copy()
    misprinted[5] = 0.0060
    mean = float(misprinted @ pub.A_BOND_VALUES)
    sd = float(np.sqrt(misprinted @ (pub.A_BOND_VALUES - mean) ** 2))
    assert misprinted.sum() == pytest.approx(1.0034)
    assert mean == pytest.approx(106.55, abs=0.01) and round(sd, 2) == 1.49
    assert 107.09 + mean == pytest.approx(213.63, abs=0.005)

    correct = value_stats(T[pub.A], pub.A_BOND_VALUES)
    assert (round(correct.mean, 2), round(correct.sd, 2)) == (106.20, 1.42)
    pair = portfolio_value_stats([T[pub.BBB], T[pub.A]], [pub.BBB_BOND_VALUES, pub.A_BOND_VALUES], [[1, 0.3], [0.3, 1]])
    assert round(pair.mean, 2) == 213.29
    # the pairwise formula agrees with the document's own joint table (printed to 0.01%)
    table = pub.JOINT_030 / pub.JOINT_030.sum()
    values = pub.BBB_BOND_VALUES[:, None] + pub.A_BOND_VALUES[None, :]
    assert float((table * values).sum()) == pytest.approx(pair.mean, abs=0.02)
    assert pair.sd == pytest.approx(3.37, abs=0.005)


def test_two_bond_exact_figures_agree_with_simulation():
    probs = [T[pub.BBB], T[pub.A]]
    values = np.array([pub.BBB_BOND_VALUES, pub.A_BOND_VALUES])
    exact = portfolio_value_stats(probs, values, [[1, 0.3], [0.3, 1]])
    sim = simulate(
        from_correlation_matrix([[1, 0.3], [0.3, 1]]),
        np.array([migration_thresholds(p) for p in probs]),
        values,
        scenarios=2_000_000,
        seed=11,
    )
    assert sim.totals.mean() == pytest.approx(exact.mean, abs=0.01)
    assert sim.totals.std() == pytest.approx(exact.sd, rel=0.01)
    assert np.quantile(sim.totals, 0.01, method="inverted_cdf") == pytest.approx(204.40)


def test_three_asset_means_and_variances_chapter_9():
    probs = [T[pub.BBB], T[pub.A], T[pub.CCC]]
    res = portfolio_value_stats(probs, pub.THREE_ASSET_VALUES, [[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]])
    assert np.round(res.asset_means, 2) == pytest.approx([4.28, 2.12, 0.97])
    assert round(res.mean, 2) == 7.38
    assert np.round(np.diag(res.covariance), 3) == pytest.approx([0.014, 0.001, 0.044])


def test_three_asset_covariances_are_checked_by_simulation_not_by_the_document():
    """Chapter 9 prints covariances (0.0015, 0.0125, 0.0030) that imply value correlations above the
    asset correlations they are derived from, which the model cannot produce. They are not used.
    The pairwise result is checked against a simulation of the same model instead."""
    probs = [T[pub.BBB], T[pub.A], T[pub.CCC]]
    corr = [[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]]
    res = portfolio_value_stats(probs, pub.THREE_ASSET_VALUES, corr)
    printed = 0.0015 / np.sqrt(0.014 * 0.001)
    assert printed > 0.3  # the printed covariance implies a value correlation above the asset correlation
    sim = simulate(
        from_correlation_matrix(corr),
        np.array([migration_thresholds(p) for p in probs]),
        pub.THREE_ASSET_VALUES,
        scenarios=2_000_000,
        seed=5,
    )
    assert sim.totals.mean() == pytest.approx(res.mean, abs=0.001)
    assert sim.totals.std() == pytest.approx(res.sd, rel=0.01)
    assert np.all(res.marginal_sds > 0) and res.marginal_sds.sum() < res.sd
