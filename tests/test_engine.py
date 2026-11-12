"""Portfolio runs: loan file and settings in, a traceable result out."""

import csv
import io
import itertools
import json

import numpy as np
import pytest

from eqlx_portfolio import InputError, config, data, engine, synthetic


def _loans(text):
    return data.from_records(csv.DictReader(io.StringIO(text)))


@pytest.fixture(scope="module")
def sample():
    loans_csv, settings_json = synthetic.generate(200, seed=7)
    raw = json.loads(settings_json)
    raw["scenarios"] = 50_000
    return _loans(loans_csv), raw


@pytest.fixture(scope="module")
def result(sample):
    loans, raw = sample
    return engine.run(loans, config.from_dict(raw))


def test_result_is_complete_and_consistent(sample, result):
    loans, _ = sample
    b = data.group(loans)
    assert result["portfolio"] == {
        "loans": len(loans),
        "borrowers": len(b),
        "sectors": 6,
        "exposure": pytest.approx(loans.exposure.sum()),
        "loss_if_all_default": pytest.approx(float(loans.exposure @ loans.lgd)),
    }
    assert result["expected_loss"] == pytest.approx(float((loans.exposure * loans.lgd) @ loans.pd))
    low, high = result["levels"]
    assert low["confidence"] == 0.99 and high["confidence"] == 0.999
    assert high["value_at_risk"] > low["value_at_risk"] > result["expected_loss"]
    for lv in result["levels"]:
        assert lv["economic_capital"] == pytest.approx(lv["value_at_risk"] - result["expected_loss"])
        assert lv["expected_shortfall"] >= lv["value_at_risk"]
        assert lv["value_at_risk_95_band"][0] <= lv["value_at_risk"] <= lv["value_at_risk_95_band"][1]
    checks = result["checks"]
    assert (
        abs(checks["expected_loss_simulated"] - checks["expected_loss_exact"])
        < 4 * checks["expected_loss_standard_error"]
    )
    assert result["loss_sd"] == pytest.approx(checks["loss_sd_exact"], rel=0.03)
    assert len(result["steps"]) >= 4 and len(result["top_contributors"]) == 10
    json.dumps(result)  # plain data all the way down


def test_risk_shares_add_up_and_show_the_planted_concentration(result):
    sectors = result["concentration"]["by_sector"]
    assert sum(s["risk_share"] for s in sectors) == pytest.approx(1.0)
    assert sum(s["exposure_share"] for s in sectors) == pytest.approx(1.0)
    assert sum(s["risk_contribution"] for s in sectors) == pytest.approx(result["levels"][-1]["expected_shortfall"])
    construction = next(s for s in sectors if s["sector"] == "construction")
    assert sectors[0] is construction and construction["risk_share"] > construction["exposure_share"]
    top = result["top_contributors"]
    assert {r["borrower_id"] for r in top[:5]} <= {f"B{i:04d}" for i in range(1, 6)} | {r["borrower_id"] for r in top}
    assert all(a["risk_contribution"] >= b["risk_contribution"] for a, b in itertools.pairwise(top))
    assert sum(r["borrower_id"] in {f"B{i:04d}" for i in range(1, 6)} for r in top[:6]) >= 4


def test_same_input_same_output(sample, result):
    loans, raw = sample
    assert engine.run(loans, config.from_dict(raw)) == result
    assert engine.run(loans, config.from_dict({**raw, "seed": 2})) != result


def test_more_correlation_means_more_capital_same_expected_loss(sample):
    loans, raw = sample
    out = []
    for rho in (0.0, 0.1, 0.3):
        s = config.from_dict({**raw, "correlation": {"model": "single", "rho": rho}})
        out.append(engine.run(loans, s))
    assert out[0]["expected_loss"] == out[1]["expected_loss"] == out[2]["expected_loss"]
    caps = [r["levels"][-1]["economic_capital"] for r in out]
    assert caps[0] < caps[1] < caps[2]
    bench = out[2]["checks"]["large_portfolio_benchmark"][-1]
    assert bench["confidence"] == 0.999 and bench["unexpected_loss"] > 0
    # a real portfolio has lumps the benchmark ignores, so it needs more than the benchmark
    assert out[2]["levels"][-1]["economic_capital"] > bench["unexpected_loss"]


def test_basel_correlation_model(sample):
    loans, raw = sample
    res = engine.run(loans, config.from_dict({**raw, "correlation": {"model": "basel_corporate"}}))
    assert "Basel corporate correlation" in res["steps"][1]
    assert len(res["checks"]["large_portfolio_benchmark"]) == 2


def test_creditriskplus_agrees_with_simulation_for_one_factor(sample):
    loans, raw = sample
    corr = {"model": "single", "rho": 0.15}
    sim = engine.run(loans, config.from_dict({**raw, "correlation": corr, "scenarios": 200_000}))
    crp = engine.run(
        loans, config.from_dict({"method": "creditriskplus", "confidence": [0.99, 0.999], "correlation": corr})
    )
    assert crp["expected_loss"] == pytest.approx(sim["expected_loss"])
    # Two different models of the same default correlation: close in the body, apart in the far tail.
    assert crp["loss_sd"] == pytest.approx(sim["loss_sd"], rel=0.10)
    assert crp["levels"][0]["value_at_risk"] == pytest.approx(sim["levels"][0]["value_at_risk"], rel=0.15)
    assert "value_at_risk_95_band" not in crp["levels"][0]
    assert crp["checks"]["probability_beyond_last_unit"] < 1e-9
    assert sum(s["risk_share"] for s in crp["concentration"]["by_sector"]) == pytest.approx(1.0)
    assert engine.run(loans, config.from_dict({"method": "creditriskplus", "correlation": corr})) is not None


def test_creditriskplus_sector_volatility_settings(sample):
    loans, raw = sample
    within = raw["correlation"]["within"]
    base = {"method": "creditriskplus", "correlation": {"model": "sector", "within": within}}
    derived = engine.run(loans, config.from_dict(base))
    assert "derived from the asset correlation" in derived["steps"][2]
    given = engine.run(loans, config.from_dict({**base, "creditriskplus": {"default_sigma": 0.0}}))
    assert given["loss_sd"] < derived["loss_sd"]
    calm = engine.run(
        loans,
        config.from_dict({**base, "creditriskplus": {"sector_sigma": {"construction": 0.0}, "loss_unit": 50_000}}),
    )
    assert calm["checks"]["loss_unit"] == 50_000 and calm["loss_sd"] < derived["loss_sd"]
    with pytest.raises(InputError, match="unknown sector"):
        engine.run(loans, config.from_dict({**base, "creditriskplus": {"sector_sigma": {"mining": 1.0}}}))


def test_what_if_uses_the_same_scenarios(sample, result):
    loans, raw = sample
    s = config.from_dict(raw)
    new = data.from_records([{"loan_id": "N1", "sector": "construction", "exposure": 30e6, "pd": 0.02, "lgd": 0.45}])
    w = engine.what_if(loans, new, s)
    last = w["levels"][-1]
    assert last["value_at_risk_before"] == result["levels"][-1]["value_at_risk"]
    assert last["expected_shortfall_before"] == pytest.approx(result["levels"][-1]["expected_shortfall"])
    assert w["expected_loss_after"] - w["expected_loss_before"] == pytest.approx(30e6 * 0.02 * 0.45)
    assert last["expected_shortfall_change"] > 0 and last["risk_contribution_of_new_loans"] > 0
    assert w["herfindahl_index_after"] != w["herfindahl_index_before"]
    assert w["new_borrowers"][0]["borrower_id"] == "N1"
    # the same loan in a small, calmer sector costs less capital
    other = data.from_records([{"loan_id": "N1", "sector": "energy", "exposure": 30e6, "pd": 0.02, "lgd": 0.45}])
    assert engine.what_if(loans, other, s)["capital_per_unit_of_new_exposure"] < w["capital_per_unit_of_new_exposure"]
    with pytest.raises(InputError, match="already in the portfolio"):
        engine.what_if(
            loans,
            data.from_records([{"loan_id": "X", "borrower_id": "B0001", "exposure": 1, "pd": 0.02, "lgd": 0.4}]),
            s,
        )
    with pytest.raises(InputError, match="already in the portfolio"):
        engine.what_if(loans, data.from_records([{"loan_id": "L00001", "exposure": 1, "pd": 0.02, "lgd": 0.4}]), s)


def test_what_if_with_creditriskplus(sample):
    loans, _ = sample
    s = config.from_dict({"method": "creditriskplus", "correlation": {"model": "single", "rho": 0.15}})
    new = data.from_records([{"loan_id": "N1", "exposure": 30e6, "pd": 0.02, "lgd": 0.45}])
    w = engine.what_if(loans, new, s)
    assert w["levels"][-1]["expected_shortfall_change"] > 0
    assert w["levels"][-1]["value_at_risk_before"] == engine.run(loans, s)["levels"][-1]["value_at_risk"]


def test_loans_of_one_borrower_default_together():
    one = _loans("loan_id,borrower_id,exposure,pd,lgd\na,X,100,0.05,0.5\nb,X,100,0.05,0.5\n")
    two = _loans("loan_id,exposure,pd,lgd\na,100,0.05,0.5\nb,100,0.05,0.5\n")
    s = config.from_dict({"correlation": {"model": "single", "rho": 0.0}, "scenarios": 200_000, "confidence": [0.99]})
    together, apart = engine.run(one, s), engine.run(two, s)
    assert together["portfolio"]["borrowers"] == 1 and apart["portfolio"]["borrowers"] == 2
    assert together["expected_loss"] == apart["expected_loss"] == 5.0
    assert together["checks"]["loss_sd_exact"] == pytest.approx(100 * np.sqrt(0.05 * 0.95))
    assert apart["checks"]["loss_sd_exact"] == pytest.approx(50 * np.sqrt(2 * 0.05 * 0.95))
    assert engine.contributions(one, s)[0]["loans"] == 2


def test_describe(sample):
    loans, _ = sample
    d = engine.describe(loans)
    assert d["loans"] == len(loans) and d["borrowers"] == 200 and len(d["sectors"]) == 6
    assert d["largest_borrowers"][0]["exposure"] >= d["largest_borrowers"][1]["exposure"]
    assert d["concentration"]["effective_number_of_borrowers"] == pytest.approx(
        1 / d["concentration"]["herfindahl_index"]
    )
    assert 0 < d["pd_exposure_weighted"] < 0.2 and not d["has_lgd_sd"]


def test_lgd_sd_column(sample):
    text = "loan_id,exposure,pd,lgd,lgd_sd\n" + "".join(f"L{i},100,0.05,0.5,0.2\n" for i in range(50))
    loans = _loans(text)
    s = config.from_dict({"correlation": {"model": "single", "rho": 0.1}, "scenarios": 100_000})
    res = engine.run(loans, s)
    fixed = engine.run(_loans(text.replace(",0.2\n", ",\n")), s)
    assert res["loss_sd"] > fixed["loss_sd"] and res["expected_loss"] == fixed["expected_loss"]
    assert res["loss_sd"] == pytest.approx(res["checks"]["loss_sd_exact"], rel=0.03)
    with pytest.raises(InputError, match="fixed loss given default"):
        engine.run(
            loans, config.from_dict({"method": "creditriskplus", "correlation": {"model": "single", "rho": 0.1}})
        )


@pytest.mark.parametrize(
    "text, message",
    [
        ("loan_id,exposure,pd\na,1,0.1\n", "missing"),
        ("loan_id,exposure,pd,lgd\na,1,0.1,0.5\na,1,0.1,0.5\n", "more than once"),
        ("loan_id,exposure,pd,lgd\na,-1,0.1,0.5\n", "exposure must be positive"),
        ("loan_id,exposure,pd,lgd\na,1,1.0,0.5\n", "pd must be"),
        ("loan_id,exposure,pd,lgd\na,1,0.1,1.5\n", "lgd must lie"),
        ("loan_id,exposure,pd,lgd\na,x,0.1,0.5\n", "not a number"),
        ("loan_id,exposure,pd,lgd\na,nan,0.1,0.5\n", "finite"),
        ("loan_id,exposure,pd,lgd,lgd_sd\na,1,0.1,0.5,0.6\n", "lgd_sd is too large"),
        ("loan_id,exposure,pd,lgd\n", "no loans"),
        ("loan_id,exposure,pd,lgd\na,1,,0.5\n", "row 2: missing pd"),
    ],
)
def test_bad_loan_rows_are_refused_with_the_row(text, message):
    with pytest.raises(InputError, match=message):
        _loans(text)


def test_borrower_consistency():
    with pytest.raises(InputError, match="different pd"):
        data.group(_loans("loan_id,borrower_id,exposure,pd,lgd\na,X,1,0.1,0.5\nb,X,1,0.2,0.5\n"))
    with pytest.raises(InputError, match="different sectors"):
        data.group(_loans("loan_id,borrower_id,sector,exposure,pd,lgd\na,X,s1,1,0.1,0.5\nb,X,s2,1,0.1,0.5\n"))


def test_csv_file_with_bom_and_missing_columns(tmp_path):
    p = tmp_path / "loans.csv"
    p.write_text("﻿loan_id, exposure ,pd,lgd\na,10,0.1,0.5\n", encoding="utf-8")
    assert len(data.read_csv(p)) == 1
    p.write_text("loan_id,exposure\na,10\n", encoding="utf-8")
    with pytest.raises(InputError, match="missing column"):
        data.read_csv(p)
    p.write_text("", encoding="utf-8")
    with pytest.raises(InputError, match="empty"):
        data.read_csv(p)
    p.write_text("loan_id,exposure,pd,lgd\n" + "a,1,0.1,0.5\n" * 1, encoding="utf-8")
    with pytest.raises(InputError, match="more than 0 loans"):
        data.read_csv(p, max_rows=0)


OK = {"correlation": {"model": "single", "rho": 0.2}}


@pytest.mark.parametrize(
    "raw, message",
    [
        ({}, "correlation block"),
        ({**OK, "mood": 1}, "unknown setting"),
        ({**OK, "method": "magic"}, "method must be"),
        ({**OK, "confidence": [1.0]}, "confidence"),
        ({**OK, "confidence": []}, "confidence"),
        ({**OK, "scenarios": 10}, "scenarios"),
        ({**OK, "scenarios": 1e5}, "scenarios"),
        ({**OK, "seed": -1}, "seed"),
        ({**OK, "seed": True}, "seed"),
        ({"correlation": {"model": "single"}}, "rho is required"),
        ({"correlation": {"model": "single", "rho": 1.0}}, "rho"),
        ({"correlation": {"model": "single", "rho": 0.2, "within": {}}}, "does not apply"),
        ({"correlation": {"model": "sector"}}, "needs correlation.within"),
        ({"correlation": {"model": "sector", "default_within": 0.2, "between": {"a": 0.1}}}, "two different sectors"),
        ({"correlation": {"model": "sector", "default_within": 0.2, "between": {"a|b": 0.1, "b|a": 0.2}}}, "twice"),
        ({"correlation": {"model": "copula"}}, "correlation.model"),
        ({**OK, "creditriskplus": {"loss_unit": 5}}, "applies only when method is creditriskplus"),
        ({**OK, "method": "creditriskplus", "seed": 3}, "apply only to simulation"),
        ({"method": "creditriskplus", "correlation": {"model": "basel_corporate"}}, "single or the sector"),
        (
            {
                "method": "creditriskplus",
                "correlation": {"model": "sector", "default_within": 0.2, "default_between": 0.5},
            },
            "independent",
        ),
        ({**OK, "method": "creditriskplus", "creditriskplus": {"loss_unit": 0}}, "loss_unit"),
        ({**OK, "top": -1}, "top"),
    ],
)
def test_bad_settings_are_refused(raw, message):
    with pytest.raises(InputError, match=message):
        config.from_dict(raw)


def test_settings_round_trip_and_overrides(sample):
    _, raw = sample
    s = config.from_dict(raw)
    assert config.from_dict(s.to_dict()) == s
    changed = config.with_overrides(raw, {"correlation.default_between": 0.2, "scenarios": 2000})
    assert changed["correlation"]["default_between"] == 0.2 and raw["correlation"]["default_between"] == 0.5
    assert config.from_dict(changed).scenarios == 2000
    with pytest.raises(InputError):
        config.with_overrides(raw, {"seed.x": 1})


def test_sector_settings_must_match_the_loan_file(sample):
    loans, raw = sample
    for corr, message in (
        ({"model": "sector", "within": {"mining": 0.2}, "default_within": 0.1}, "not in the loan file"),
        ({"model": "sector", "within": {"construction": 0.2}}, "no within-sector correlation"),
        ({"model": "sector", "default_within": 0.1, "between": {"mining|energy": 0.2}}, "not in the loan file"),
    ):
        with pytest.raises(InputError, match=message):
            engine.run(loans, config.from_dict({**raw, "correlation": corr}))
