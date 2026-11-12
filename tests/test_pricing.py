"""Loan pricing against closed forms, hand arithmetic and the published scheme it follows."""

import json

import numpy as np
import pytest

from eqlx_portfolio import (
    InputError,
    capital_from_basel_formula,
    capital_from_risk_weight,
    config,
    data,
    engine,
    large_portfolio_loss,
    price_loan,
    synthetic,
)
from eqlx_portfolio.pricing import discount_factors, survival

BASE = dict(notional=1_000_000.0, years=5, pd=0.02, lgd=0.45, capital=0.08, target_return=0.12, funding=0.05)


@pytest.mark.parametrize("years, recovery, pd", [(1, 0.4, 0.03), (5, 0.4, 0.03), (10, 0.2, 0.01), (7, 0.0, 0.10)])
def test_expected_loss_margin_matches_the_published_closed_form(years, recovery, pd):
    """Engelmann and Pham (2020), eq. 28: a bullet loan with yearly interest, zero interest rates and a
    constant recovery rate R has expected loss margin (1 - R)(1 - v(m)) / sum of v(i), v = survival."""
    v = (1 - pd) ** np.arange(1, years + 1)
    p = price_loan(notional=1.0, years=years, pd=pd, lgd=1 - recovery, capital=0.08, target_return=0.1, funding=0.0)
    assert p.funding_cost_rate == pytest.approx(0.0, abs=1e-15)
    assert p.expected_loss_margin == pytest.approx((1 - recovery) * (1 - v[-1]) / v.sum(), rel=1e-12)


def test_one_year_loan_by_hand():
    # Lend 1 for a year at rate y, funded at f. Survive (prob v): get 1 + y. Default: get R.
    # Break-even without costs: (1 + y) v + R (1 - v) = 1 + f.
    f, pd, recovery = 0.30, 0.04, 0.35
    v = 1 - pd
    y = ((1 + f) - recovery * (1 - v)) / v - 1
    p = price_loan(notional=1.0, years=1, pd=pd, lgd=1 - recovery, capital=0.1, target_return=0.2, funding=f)
    assert p.funding_cost_rate == pytest.approx(f)
    assert p.funding_cost_rate + p.expected_loss_margin == pytest.approx(y, rel=1e-12)
    assert p.first_year_pd == pytest.approx(pd) and p.average_life == 1.0


def test_without_default_risk_the_hurdle_is_the_sum_of_the_inputs():
    p = price_loan(**{**BASE, "pd": 0.0, "operating_cost": 0.01, "capital_return": 0.03, "tax_rate": 0.25})
    assert p.expected_loss_margin == pytest.approx(0.0, abs=1e-15)
    assert p.operating_cost_margin == pytest.approx(0.01)
    assert p.capital_margin == pytest.approx((0.12 - 0.03) * 0.08 / 0.75)
    assert p.hurdle_rate == pytest.approx(0.05 + 0.01 + 0.0096)


@pytest.mark.parametrize(
    "repayment, per_year, tax", [("bullet", 1, 0.0), ("equal_principal", 4, 0.3), ("bullet", 12, 0.2)]
)
def test_at_the_hurdle_rate_raroc_is_the_target(repayment, per_year, tax):
    terms = {**BASE, "repayment": repayment, "payments_per_year": per_year, "tax_rate": tax, "operating_cost": 0.008}
    hurdle = price_loan(**terms).hurdle_rate
    at = price_loan(**terms, rate=hurdle)
    assert at.raroc == pytest.approx(0.12, abs=1e-12) and at.yearly_value_added == pytest.approx(0.0, abs=1e-6)
    above = price_loan(**terms, rate=hurdle + 0.01)
    assert above.raroc == pytest.approx(0.12 + (1 - tax) * 0.01 / 0.08)
    assert above.yearly_value_added == pytest.approx((1 - tax) * 0.01 * 1_000_000)
    d = above.to_dict()
    assert d["meets_target"] and d["rate_minus_hurdle"] == pytest.approx(0.01)
    assert not price_loan(**terms, rate=hurdle - 0.01).to_dict()["meets_target"]
    assert "raroc" not in price_loan(**terms).to_dict()
    json.dumps(d, allow_nan=False)


def test_more_risk_capital_or_cost_raises_the_hurdle():
    base = price_loan(**BASE).hurdle_rate
    for change in (
        {"pd": 0.05},
        {"lgd": 0.8},
        {"capital": 0.16},
        {"operating_cost": 0.01},
        {"funding": 0.06},
        {"tax_rate": 0.3},
    ):
        assert price_loan(**{**BASE, **change}).hurdle_rate > base, change
    assert price_loan(**{**BASE, "capital_return": 0.04}).hurdle_rate < base


def test_funding_follows_the_loan_convention_and_the_curve():
    # a flat funding rate quoted like the loan's rate gives exactly that funding cost, at any frequency
    for per_year in (1, 2, 4, 12):
        p = price_loan(**{**BASE, "funding": 0.40, "payments_per_year": per_year})
        assert p.funding_cost_rate == pytest.approx(0.40, rel=1e-12)
    curve = [[1, 0.03], [5, 0.05], [10, 0.07]]
    bullet = price_loan(**{**BASE, "years": 10, "funding": curve})
    amortising = price_loan(**{**BASE, "years": 10, "funding": curve, "repayment": "equal_principal"})
    assert 0.05 < bullet.funding_cost_rate < 0.07
    assert amortising.funding_cost_rate < bullet.funding_cost_rate  # shorter average life on a rising curve
    assert amortising.average_life == pytest.approx(5.5) and bullet.average_life == 10
    t = np.array([0.5, 1.0, 3.0, 12.0])
    assert discount_factors(t, curve) == pytest.approx((1 + np.array([0.03, 0.03, 0.04, 0.07])) ** -t)


def test_survival_curve():
    t = np.array([0.25, 1.0, 2.5, 4.0])
    assert survival(t, 0.02) == pytest.approx(0.98**t)
    cumulative = [0.02, 1 - 0.98**2, 1 - 0.98**3]
    assert survival(t, cumulative) == pytest.approx(0.98**t)
    rising = survival(np.array([1.0, 2.0, 3.0]), [0.01, 0.04, 0.10])
    assert rising == pytest.approx([0.99, 0.96, 0.90])
    for bad in ([0.05, 0.04], [0.5, 1.0], [-0.1]):
        with pytest.raises(InputError):
            survival(t, bad)


def test_collateral_follows_equation_11():
    # recovery = min(1, (collateral + unsecured recovery x uncovered part) / outstanding)
    p = price_loan(**{**BASE, "lgd": None, "collateral": 600_000.0, "unsecured_recovery": 0.2})
    assert p.loss_given_default_at_start == pytest.approx(1 - (600_000 + 0.2 * 400_000) / 1_000_000)
    covered = price_loan(**{**BASE, "lgd": None, "collateral": 1_200_000.0, "unsecured_recovery": 0.0})
    # fully covered principal still loses the interest of the period in which the borrower defaults
    assert 0 < covered.expected_loss_margin < 0.0015 < p.expected_loss_margin
    bullet = price_loan(**{**BASE, "lgd": None, "collateral": 600_000.0, "unsecured_recovery": 0.2, "years": 10})
    paid_down = price_loan(
        **{
            **BASE,
            "lgd": None,
            "collateral": 600_000.0,
            "unsecured_recovery": 0.2,
            "years": 10,
            "repayment": "equal_principal",
        }
    )
    assert paid_down.expected_loss_margin < bullet.expected_loss_margin  # the same collateral covers more of less


def test_published_example_is_reproduced_approximately():
    """Engelmann and Pham (2020), section 6, tables 3: four ten-year loans at 4%, quarterly interest,
    rating grade 3, capital 8%.

    The article builds its discount curve from swap and basis-swap quotes, which this tool does not
    rebuild. With a flat curve at the par rate the article prints (ys + sf + sb) the margins come out
    within 0.04 percentage points and RAROC within 0.4 points of the printed values. That is a
    plausibility check, not a match; the exact checks are the closed forms above.
    """
    pd_one_year = 1 - np.exp(-np.exp(-5.0 + 0.04 * 10.0))  # the article's hazard model, grade 3, rate 4%
    assert pd_one_year == pytest.approx(0.01, abs=1e-4)
    printed = {  # loan: (par rate %, expected loss margin %, cost margin %, RAROC %)
        "I": (2.14, 0.29, 0.52, 12.94),
        "II": (2.14, 0.78, 0.52, 6.88),
        "III": (1.93, 0.16, 0.52, 17.28),
        "IV": (1.93, 0.78, 0.52, 9.51),
    }
    loans = {
        "I": dict(collateral=600_000.0, unsecured_recovery=0.2, repayment="bullet"),
        "II": dict(lgd=0.8, repayment="bullet"),
        "III": dict(collateral=600_000.0, unsecured_recovery=0.2, repayment="equal_principal"),
        "IV": dict(lgd=0.8, repayment="equal_principal"),
    }
    for name, (par, el, cost, raroc) in printed.items():
        p = price_loan(
            notional=1_000_000.0, years=10, pd=pd_one_year, capital=0.08, target_return=0.10, funding=par / 100,
            operating_cost=0.005, payments_per_year=4, rate=0.04, **loans[name],
        )  # fmt: skip
        assert p.funding_cost_rate * 100 == pytest.approx(par, abs=0.02 if name in ("I", "II") else 0.25)
        assert p.operating_cost_margin * 100 == pytest.approx(cost, abs=0.01)
        if name in ("I", "II"):
            assert p.expected_loss_margin * 100 == pytest.approx(el, abs=0.04)
            assert p.raroc * 100 == pytest.approx(raroc, abs=0.4)
    # the equal_principal loans of the article repay 5% a year and a balloon; this tool repays in full,
    # so only their cost margin and funding level are compared


def test_capital_helpers():
    assert capital_from_risk_weight(1.0, 0.12) == pytest.approx(0.12)
    assert capital_from_risk_weight(0.5, 0.08) == pytest.approx(0.04)
    k = capital_from_basel_formula(0.01, 0.45, asset_correlation=0.2)
    assert k == pytest.approx(large_portfolio_loss([0.45], [0.01], 0.2, 0.999) - 0.0045)
    assert capital_from_basel_formula(0.01, 0.45) > capital_from_basel_formula(0.01, 0.45, asset_correlation=0.05)
    with pytest.raises(InputError):
        capital_from_basel_formula(0.0, 0.45)


@pytest.mark.parametrize(
    "change",
    [
        {"notional": 0}, {"years": 0}, {"years": 2.3}, {"payments_per_year": 3}, {"repayment": "annuity"},
        {"capital": 0}, {"capital": 1.5}, {"pd": 1.0}, {"pd": True}, {"lgd": 1.2}, {"tax_rate": 1.0},
        {"funding": -1.0}, {"funding": [[2, 0.05], [1, 0.04]]}, {"funding": "high"}, {"operating_cost": -0.1},
        {"lgd": None}, {"collateral": 100.0}, {"target_return": float("nan")}, {"rate": 10**400},
        {"years": 500, "payments_per_year": 12},
    ],
)  # fmt: skip
def test_bad_input_is_refused(change):
    with pytest.raises(InputError):
        price_loan(**{**BASE, **change})


@pytest.fixture(scope="module")
def book():
    loans_csv, settings_json = synthetic.generate(150, seed=7)
    raw = json.loads(settings_json)
    raw["scenarios"] = 50_000
    return data.from_text(loans_csv), config.from_dict(raw)


TERMS = {"years": 3, "funding": 0.40, "target_return": 0.45, "operating_cost": 0.01, "capital_return": 0.38,
         "tax_rate": 0.30, "payments_per_year": 4, "rate": 0.43, "risk_weight": 1.0, "capital_ratio": 0.12}  # fmt: skip


def _new(sector, exposure=30e6):
    return data.from_records([{"loan_id": "N1", "sector": sector, "exposure": exposure, "pd": 0.02, "lgd": 0.45}])


def test_the_same_loan_costs_more_where_the_book_is_heavy(book):
    loans, settings = book
    heavy = engine.price_new_loan(loans, _new("construction"), settings, TERMS)
    light = engine.price_new_loan(loans, _new("energy"), settings, TERMS)
    h = {v["capital_basis"]: v for v in heavy["views"]}
    lt = {v["capital_basis"]: v for v in light["views"]}
    assert set(h) == {"portfolio", "basel_formula", "risk_weight"}
    assert h["portfolio"]["capital_per_unit_lent"] > lt["portfolio"]["capital_per_unit_lent"]
    assert h["portfolio"]["hurdle_rate"] > lt["portfolio"]["hurdle_rate"]
    assert h["portfolio"]["raroc"] < lt["portfolio"]["raroc"]
    # the formula and the risk weight cannot see the book: they price both loans the same
    for basis in ("basel_formula", "risk_weight"):
        assert h[basis]["hurdle_rate"] == lt[basis]["hurdle_rate"]
    assert h["risk_weight"]["capital_per_unit_lent"] == pytest.approx(0.12)
    # everything but the capital margin is the same in every view
    for key in ("funding_cost_rate", "expected_loss_margin", "operating_cost_margin"):
        assert len({round(v["components"][key], 12) for v in heavy["views"]}) == 1
    assert h["portfolio"]["components"]["funding_cost_rate"] == pytest.approx(0.40)
    json.dumps(heavy, allow_nan=False)


def test_portfolio_capital_is_the_what_if_rise_beyond_expected_loss(book):
    loans, settings = book
    priced = engine.price_new_loan(loans, _new("construction"), settings, TERMS)
    w = engine.what_if(loans, _new("construction"), settings)
    expected = (w["levels"][-1]["expected_shortfall_change"] - 30e6 * 0.02 * 0.45) / 30e6
    assert priced["views"][0]["capital_per_unit_lent"] == pytest.approx(expected)
    assert priced["portfolio_effect"]["expected_shortfall_change"] == w["levels"][-1]["expected_shortfall_change"]
    direct = price_loan(
        notional=30e6, years=3, pd=0.02, lgd=0.45, capital=expected, target_return=0.45, funding=0.40,
        operating_cost=0.01, capital_return=0.38, tax_rate=0.30, payments_per_year=4, rate=0.43,
    )  # fmt: skip
    assert priced["views"][0]["hurdle_rate"] == pytest.approx(direct.hurdle_rate)
    assert priced["views"][0]["raroc"] == pytest.approx(direct.raroc)


def test_pricing_with_creditriskplus_and_without_optional_terms(book):
    loans, _ = book
    s = config.from_dict({"method": "creditriskplus", "correlation": {"model": "single", "rho": 0.15}})
    priced = engine.price_new_loan(loans, _new("energy"), s, {"years": 2, "funding": 0.05, "target_return": 0.12})
    assert [v["capital_basis"] for v in priced["views"]] == ["portfolio", "basel_formula"]
    assert all("raroc" not in v and v["hurdle_rate"] > 0.05 for v in priced["views"])


def test_pricing_terms_are_checked(book):
    loans, settings = book
    two = data.from_records([{"loan_id": f"N{i}", "exposure": 1e6, "pd": 0.02, "lgd": 0.45} for i in range(2)])
    cases = [
        (_new("energy"), {**TERMS, "mood": 1}, "unknown term"),
        (_new("energy"), {"years": 3}, "terms need"),
        (_new("energy"), {k: v for k, v in TERMS.items() if k != "capital_ratio"}, "go together"),
        (two, TERMS, "one new loan"),
        (_new("energy"), [1], "must be an object"),
        (_new("energy"), {**TERMS, "years": 2.3}, "payment periods"),
        (
            data.from_records([{"loan_id": "N1", "sector": "energy", "exposure": 1e6, "pd": 0.0, "lgd": 0.45}]),
            TERMS,
            "positive pd",
        ),
    ]
    for new, terms, message in cases:
        with pytest.raises(InputError, match=message):
            engine.price_new_loan(loans, new, settings, terms)


def test_a_loan_too_small_to_measure_is_reported_not_priced(book):
    loans, settings = book
    tiny = engine.price_new_loan(loans, _new("energy", exposure=1.0), settings, TERMS)
    first = tiny["views"][0]
    assert first["capital_basis"] == "portfolio"
    assert ("not_priced" in first) == (first["capital_per_unit_lent"] <= 0)
    assert "hurdle_rate" in tiny["views"][1]
