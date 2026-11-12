"""Regressions: defects found by an independent review, each pinned by a test."""

import json

import numpy as np
import pytest

from eqlx_portfolio import (
    InputError,
    cli,
    config,
    creditriskplus,
    data,
    default_thresholds,
    engine,
    joint_migration,
    migration_thresholds,
    portfolio_value_stats,
    quantiles,
    simulate,
    single_factor,
    synthetic,
)


@pytest.fixture(scope="module")
def book():
    loans_csv, settings_json = synthetic.generate(150, seed=7)
    raw = json.loads(settings_json)
    raw["scenarios"] = 20_000
    return data.from_text(loans_csv), raw


def test_zero_probability_best_states_give_no_nan():
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = np.concatenate(([0.0], rng.dirichlet(np.ones(6))))
        z = migration_thresholds(p)
        assert not np.any(np.isnan(z)) and z[0] == np.inf
        j = joint_migration(p, [0.9, 0.1], 0.3)
        assert not np.any(np.isnan(j)) and j[0].sum() == 0.0 and j.sum() == pytest.approx(1.0)
    ccc = [0, 0, 0.0022, 0.013, 0.0238, 0.1124, 0.6486, 0.20]
    assert not np.any(np.isnan(migration_thresholds(ccc)))
    res = portfolio_value_stats([ccc, [0.9, 0.1]], [[8, 7, 6, 5, 4, 3, 2, 1], [1, 0]], [[1, 0.3], [0.3, 1]])
    assert np.isfinite(res.sd)


def test_what_if_risk_share_is_a_share_of_the_whole_portfolio(book):
    loans, raw = book
    s = config.from_dict(raw)
    new = data.from_records(
        [
            {"loan_id": "N1", "sector": "construction", "exposure": 30e6, "pd": 0.02, "lgd": 0.45},
            {"loan_id": "N2", "sector": "energy", "exposure": 100, "pd": 0.02, "lgd": 0.45},
        ]
    )
    w = engine.what_if(loans, new, s)
    es = w["levels"][-1]["expected_shortfall_after"]
    rows = {r["borrower_id"]: r for r in w["new_borrowers"]}
    assert set(rows) == {"N1", "N2"}
    for r in rows.values():
        assert r["risk_share"] == pytest.approx(r["risk_contribution"] / es)
    assert 0.02 < rows["N1"]["risk_share"] < 0.6
    assert sum(r["risk_contribution"] for r in rows.values()) == pytest.approx(
        w["levels"][-1]["risk_contribution_of_new_loans"]
    )


@pytest.mark.parametrize(
    "new",
    [
        {"loan_id": "N1", "sector": "agriculture", "exposure": 30e6, "pd": 0.02, "lgd": 0.45},
        {"loan_id": "N1", "sector": "zinc", "exposure": 30e6, "pd": 0.02, "lgd": 0.45},
        {"loan_id": "N1", "sector": "energy", "exposure": 30e6, "pd": 0.02, "lgd": 0.45, "lgd_sd": 0.2},
    ],
)
def test_what_if_before_equals_a_separate_run_even_with_a_new_sector_or_lgd_sd(book, new):
    loans, raw = book
    raw = {**raw, "correlation": {**raw["correlation"], "default_within": 0.2}}
    base = engine.run(loans, config.from_dict(raw))
    w = engine.what_if(loans, data.from_records([new]), config.from_dict(raw))
    for before, level in zip(base["levels"], w["levels"], strict=True):
        assert level["value_at_risk_before"] == before["value_at_risk"]
        assert level["expected_shortfall_before"] == pytest.approx(before["expected_shortfall"], rel=1e-12)
        assert level["expected_shortfall_change"] > 0


def test_what_if_creditriskplus_accepts_a_new_sector_named_in_the_settings(book):
    loans, raw = book
    within = {**raw["correlation"]["within"], "agriculture": 0.2}
    s = config.from_dict({"method": "creditriskplus", "correlation": {"model": "sector", "within": within}})
    new = data.from_records([{"loan_id": "N1", "sector": "agriculture", "exposure": 30e6, "pd": 0.02, "lgd": 0.45}])
    w = engine.what_if(loans, new, s)
    assert w["levels"][-1]["expected_shortfall_change"] > 0


def test_creditriskplus_default_unit_does_not_distort_a_skewed_book():
    amount = np.concatenate(([1e8], np.full(10_000, 1e4)))
    pd = np.concatenate(([1e-4], np.full(10_000, 0.05)))
    weights = np.ones((amount.size, 1))
    auto = creditriskplus(amount, pd, weights, [0.0], confidence=[0.999])
    fine = creditriskplus(amount, pd, weights, [0.0], loss_unit=1e4, confidence=[0.999])
    assert fine.sd == pytest.approx(fine.sd_before_rounding, rel=1e-9)
    assert auto.sd == pytest.approx(auto.sd_before_rounding, rel=0.006)
    assert auto.tails[0].value_at_risk == pytest.approx(fine.tails[0].value_at_risk, rel=0.02)
    # a high-pd loan of about one and a half default units no longer fails without a unit being set
    assert creditriskplus([200, 1.49], [0.01, 0.9], np.ones((2, 1)), [0.5]).loss_unit <= 1.0


def test_creditriskplus_rounding_warning_reaches_the_result():
    text = "loan_id,exposure,pd,lgd\nBIG,100000000,0.0001,1\n" + "".join(f"S{i},10000,0.05,1\n" for i in range(2000))
    s = {"method": "creditriskplus", "correlation": {"model": "single", "rho": 0.0}}
    coarse = engine.run(data.from_text(text), config.from_dict({**s, "creditriskplus": {"loss_unit": 500_000}}))
    assert any(step.startswith("Warning: rounding to loss units") for step in coarse["steps"])
    assert coarse["loss_sd"] > 1.2 * coarse["checks"]["loss_sd_before_rounding"]
    auto = engine.run(data.from_text(text), config.from_dict(s))
    assert not any(step.startswith("Warning") for step in auto["steps"])


@pytest.mark.parametrize("unit", [100, 1e-9])
def test_too_small_loss_unit_is_refused_cleanly(book, unit):
    loans, _ = book
    s = {
        "method": "creditriskplus",
        "correlation": {"model": "single", "rho": 0.1},
        "creditriskplus": {"loss_unit": unit},
    }
    with pytest.raises(InputError, match="loss_unit"):
        engine.run(loans, config.from_dict(s))


def test_files_that_are_not_utf8_are_refused_with_a_reason(tmp_path, capsys):
    loans = tmp_path / "cp1254.csv"
    loans.write_bytes(b"loan_id,exposure,pd,lgd,sector\nL1,1000,0.02,0.45,in\xfeaat\n")
    with pytest.raises(InputError, match="UTF-8"):
        data.read_csv(loans)
    settings = tmp_path / "s.json"
    settings.write_bytes(b'{"correlation": {"model": "single", "rho": 0.1}, "x": "\xfe"}')
    with pytest.raises(InputError, match="UTF-8"):
        config.read_json(settings)
    assert cli.main(["describe", str(loans)]) == 2
    assert "UTF-8" in capsys.readouterr().err


def test_band_has_no_upper_end_when_the_tail_is_too_thin(book):
    totals = np.random.default_rng(1).exponential(size=1000)
    q = quantiles(totals, [0.99, 0.999])
    assert q[0].ci_high is not None and q[0].ci_low <= q[0].value <= q[0].ci_high
    assert q[1].ci_high is None
    loans, raw = book
    res = engine.run(loans, config.from_dict({**raw, "scenarios": 1000}))
    assert res["levels"][-1]["value_at_risk_95_band"][1] is None
    assert any(step.startswith("Warning: 1000 scenarios leave only 2 outcome") for step in res["steps"])
    assert "unbounded" in cli._text(res) and "! Warning" in cli._text(res)
    json.loads(json.dumps(res, allow_nan=False))


def test_results_are_strict_json(book):
    loans, raw = book
    for s in (raw, {"method": "creditriskplus", "correlation": {"model": "single", "rho": 0.1}}):
        json.dumps(engine.run(loans, config.from_dict(s)), allow_nan=False)
    json.dumps(engine.describe(loans), allow_nan=False)


def test_huge_and_odd_numbers_are_refused():
    with pytest.raises(InputError):
        config.from_dict({"correlation": {"model": "single", "rho": 10**400}})
    with pytest.raises(InputError):
        migration_thresholds([1e308, -1e308])
    with pytest.raises(InputError):
        data.from_records([{"loan_id": "a", "exposure": 10**400, "pd": 0.1, "lgd": 0.5}])
    with pytest.raises(InputError, match="unknown column"):
        data.from_records([{"loan_id": "a", "exposure": 1, "pd": 0.1, "lgd": 0.5, "sektor": "x"}])
    with pytest.raises(InputError):
        data.from_records([{"loan_id": "a", "exposure": True, "pd": 0.1, "lgd": 0.5}])


def test_pair_with_zero_pd():
    res = engine.pair(0.0, 0.05, 0.2)
    assert res["joint_default_probability"] == 0.0 and res["default_correlation"] is None
    assert engine.pair(0.02, 0.05, 0.2)["default_correlation"] > 0


def test_simulation_side_factors_are_validated():
    m = single_factor(0.2, 4)
    out = np.column_stack((np.zeros(4), np.ones(4)))
    flag = np.array([False, False, False, True])
    with pytest.raises(InputError):
        simulate(m, default_thresholds(np.full(4, 0.1)), out, scenarios=1000, seed=1, exclude=flag, side_factors=1)
