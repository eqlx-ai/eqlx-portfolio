"""What a loan has to earn: cost components, hurdle rate and RAROC of a fixed-rate loan.

RAROC (risk-adjusted return on capital) is the loan's yearly margin after funding, expected
loss and operating costs, divided by the capital the loan ties up. The hurdle rate is the
interest rate at which RAROC equals the bank's target return.

The scheme follows Engelmann and Pham (2020), "A RAROC valuation scheme for loans and its
application in loan origination" (Risks 8(2), 63), equations 1, 11, 20-26, for a fixed-rate loan
paid out in one tranche and funded on a given discount curve:

- funding cost rate: the rate at which the loan's cash flows, discounted on the bank's funding
  curve, are worth exactly the amount lent;
- expected loss margin: what must be added so that this still holds when cash flows only arrive
  while the borrower survives and a default pays the recovery;
- operating cost margin: the yearly cost, carried by the borrowers who survive;
- capital margin: the target return on the capital tied up, less what that capital earns anyway.

The cost components are those a supervisor expects loan pricing to reflect (EBA guidelines on
loan origination and monitoring, EBA/GL/2020/06, paragraph 202): cost of capital, cost of
funding, operating costs, credit risk costs and tax. Nothing is estimated: every one of them is
an input. The tool states the break-even rate; it does not recommend a price.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._checks import Array, as_1d
from .errors import InputError
from .vasicek import basel_corporate_correlation, conditional_pd

REPAYMENTS = ("bullet", "equal_principal")
MAX_PERIODS = 1200


def _number(
    name: str, value: Any, low: float, high: float, *, open_low: bool = False, open_high: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a number")
    try:
        v = float(value)
    except OverflowError:
        raise InputError(f"{name} is outside its allowed range") from None
    if not np.isfinite(v) or v < low or v > high or (open_low and v == low) or (open_high and v == high):
        raise InputError(f"{name} is outside its allowed range")
    return v


def discount_factors(times: Array, funding: float | Sequence[Sequence[float]], payments_per_year: int = 1) -> Array:
    """Discount factors on the bank's funding curve.

    ``funding`` is one rate for all maturities, or a list of ``[years, rate]`` points; rates
    between points are interpolated linearly and held flat beyond the first and last point.
    Funding rates are quoted like the loan's own rate: per year, paid ``payments_per_year`` times
    a year. A bullet loan funded at a flat 40% therefore has a funding cost rate of exactly 40%.
    """
    if isinstance(funding, (int, float)) and not isinstance(funding, bool):
        rate = np.full(times.size, _number("funding rate", funding, -0.99, 100.0, open_low=True))
    else:
        try:
            pts = np.asarray(funding, dtype=np.float64)
        except (TypeError, ValueError):
            raise InputError("funding must be a rate or a list of [years, rate] points") from None
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0 or not np.all(np.isfinite(pts)):
            raise InputError("funding must be a rate or a list of [years, rate] points")
        if np.any(pts[:, 0] <= 0.0) or np.any(np.diff(pts[:, 0]) <= 0.0):
            raise InputError("funding curve maturities must be positive and increasing")
        if np.any(pts[:, 1] <= -0.99) or np.any(pts[:, 1] > 100.0):
            raise InputError("funding curve rates are outside the allowed range")
        rate = np.interp(times, pts[:, 0], pts[:, 1])
    out: Array = (1.0 + rate / payments_per_year) ** (-times * payments_per_year)
    return out


def survival(times: Array, pd: float | Sequence[float]) -> Array:
    """Probability that the borrower has not defaulted by each time.

    ``pd`` is a one-year default probability applied at a constant rate, or a list of cumulative
    default probabilities at the end of year 1, 2, ...; between year ends the default rate is
    constant, and beyond the last year the last year's rate continues.
    """
    if isinstance(pd, (int, float)) and not isinstance(pd, bool):
        p = _number("pd", pd, 0.0, 1.0, open_high=True)
        out: Array = (1.0 - p) ** times
        return out
    cum = as_1d("pd", pd)
    if np.any(cum < 0.0) or np.any(cum >= 1.0) or np.any(np.diff(cum) < 0.0):
        raise InputError("cumulative default probabilities must lie in [0, 1) and must not fall")
    log_s = np.concatenate(([0.0], np.log1p(-cum)))
    years = np.arange(cum.size + 1, dtype=np.float64)
    last_slope = log_s[-1] - log_s[-2]
    inside = np.interp(np.minimum(times, years[-1]), years, log_s)
    out = np.exp(inside + np.maximum(times - years[-1], 0.0) * last_slope)
    return out


@dataclass(frozen=True)
class LoanPrice:
    funding_cost_rate: float
    expected_loss_margin: float
    operating_cost_margin: float
    capital_margin: float
    hurdle_rate: float
    capital: float  # per unit lent
    target_return: float
    rate: float | None
    raroc: float | None
    yearly_value_added: float | None  # (RAROC - target) x capital, in currency a year
    average_life: float
    first_year_pd: float
    loss_given_default_at_start: float

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "components": {
                "funding_cost_rate": self.funding_cost_rate,
                "expected_loss_margin": self.expected_loss_margin,
                "operating_cost_margin": self.operating_cost_margin,
                "capital_margin": self.capital_margin,
            },
            "hurdle_rate": self.hurdle_rate,
            "capital_per_unit_lent": self.capital,
            "target_return": self.target_return,
            "average_life_years": self.average_life,
            "first_year_pd": self.first_year_pd,
            "loss_given_default_at_start": self.loss_given_default_at_start,
        }
        if self.rate is not None:
            out["rate"] = self.rate
            out["raroc"] = self.raroc
            out["meets_target"] = bool(self.raroc is not None and self.raroc >= self.target_return - 1e-12)
            out["rate_minus_hurdle"] = self.rate - self.hurdle_rate
            out["yearly_value_added"] = self.yearly_value_added
        return out


def price_loan(
    *,
    notional: float,
    years: float,
    pd: float | Sequence[float],
    capital: float,
    target_return: float,
    funding: float | Sequence[Sequence[float]],
    lgd: float | None = None,
    collateral: float | None = None,
    unsecured_recovery: float | None = None,
    operating_cost: float = 0.0,
    capital_return: float = 0.0,
    tax_rate: float = 0.0,
    payments_per_year: int = 1,
    repayment: str = "bullet",
    rate: float | None = None,
) -> LoanPrice:
    """Cost components, hurdle rate and (given ``rate``) RAROC of a fixed-rate loan.

    All rates are yearly decimals, paid ``payments_per_year`` times a year; ``funding`` is quoted
    the same way. ``capital`` is the capital tied up per unit lent (0.08 for 8%),
    from whichever source you trust: a portfolio run, the Basel formula, a risk weight.
    ``target_return`` and ``capital_return`` are after tax; ``capital_return`` is what the capital
    earns where it is invested. Loss in default is either ``lgd`` (a share of what is outstanding)
    or ``collateral`` (a cash value) with ``unsecured_recovery`` for the rest. Interest is paid
    ``payments_per_year`` times a year; principal at the end (``bullet``) or in equal parts with
    every payment (``equal_principal``). Default is recognised at payment dates.
    """
    n = _number("notional", notional, 0.0, 1e15, open_low=True)
    if isinstance(payments_per_year, bool) or payments_per_year not in (1, 2, 4, 12):
        raise InputError("payments_per_year must be 1, 2, 4 or 12")
    if repayment not in REPAYMENTS:
        raise InputError(f"repayment must be one of: {', '.join(REPAYMENTS)}")
    life = _number("years", years, 0.0, 100.0, open_low=True)
    periods = round(life * payments_per_year)
    if periods < 1 or abs(periods - life * payments_per_year) > 1e-9 or periods > MAX_PERIODS:
        raise InputError("years must be a whole number of payment periods")
    cap = _number("capital", capital, 0.0, 1.0, open_low=True)
    target = _number("target_return", target_return, -1.0, 10.0)
    cap_return = _number("capital_return", capital_return, -1.0, 10.0)
    tax = _number("tax_rate", tax_rate, 0.0, 1.0, open_high=True)
    cost = _number("operating_cost", operating_cost, 0.0, 1.0)
    offered = None if rate is None else _number("rate", rate, -1.0, 100.0)

    tau = 1.0 / payments_per_year
    times: Array = tau * np.arange(1, periods + 1, dtype=np.float64)
    if repayment == "bullet":
        repay = np.zeros(periods)
        repay[-1] = n
    else:
        repay = np.full(periods, n / periods)
    outstanding = n - np.concatenate(([0.0], np.cumsum(repay)[:-1]))  # during each period

    if (lgd is None) == (collateral is None and unsecured_recovery is None):
        raise InputError("give either lgd, or collateral with unsecured_recovery")
    if lgd is not None:
        recovery = np.full(periods, 1.0 - _number("lgd", lgd, 0.0, 1.0))
    else:
        coll = _number("collateral", 0.0 if collateral is None else collateral, 0.0, 1e15)
        unsec = _number("unsecured_recovery", 0.0 if unsecured_recovery is None else unsecured_recovery, 0.0, 1.0)
        recovery = np.minimum(1.0, (coll + unsec * np.maximum(outstanding - coll, 0.0)) / outstanding)

    disc = discount_factors(times, funding, payments_per_year)
    alive = survival(times, pd)
    alive_before = np.concatenate(([1.0], alive[:-1]))

    interest_base = float((outstanding * tau * disc).sum())
    funding_rate = (n - float((repay * disc).sum())) / interest_base
    default_value = float((outstanding * recovery * disc * (alive_before - alive)).sum())
    risky_base = float((outstanding * tau * disc * alive).sum())
    if risky_base <= 0.0:
        raise InputError("the borrower cannot be expected to survive a single payment date")
    risky_rate = (n - float((repay * disc * alive).sum()) - default_value) / risky_base
    el_margin = risky_rate - funding_rate
    cost_margin = cost * interest_base / risky_base
    capital_margin = (target - cap_return) * cap / (1.0 - tax)
    hurdle = funding_rate + el_margin + cost_margin + capital_margin

    raroc = value_added = None
    if offered is not None:
        raroc = (1.0 - tax) * (offered - funding_rate - el_margin - cost_margin) / cap + cap_return
        value_added = (raroc - target) * cap * n
    return LoanPrice(
        funding_cost_rate=funding_rate,
        expected_loss_margin=el_margin,
        operating_cost_margin=cost_margin,
        capital_margin=capital_margin,
        hurdle_rate=hurdle,
        capital=cap,
        target_return=target,
        rate=offered,
        raroc=raroc,
        yearly_value_added=value_added,
        average_life=float((repay * times).sum() / n),
        first_year_pd=float(1.0 - survival(np.array([1.0]), pd)[0]),
        loss_given_default_at_start=float(1.0 - recovery[0]),
    )


def capital_from_risk_weight(risk_weight: float, capital_ratio: float) -> float:
    """Capital per unit lent under a fixed risk weight: risk weight times the capital ratio held."""
    return _number("risk_weight", risk_weight, 0.0, 12.5) * _number("capital_ratio", capital_ratio, 0.0, 1.0)


def capital_from_basel_formula(
    pd: float, lgd: float, *, asset_correlation: float | None = None, confidence: float = 0.999
) -> float:
    """Capital per unit lent from the one-factor formula behind the Basel risk weights.

    Loss in the bad year less expected loss, with the Basel corporate correlation unless another
    is given. No maturity adjustment and no supervisory scaling: a yardstick, not a filing.
    """
    p = _number("pd", pd, 0.0, 1.0, open_low=True, open_high=True)
    g = _number("lgd", lgd, 0.0, 1.0)
    rho = (
        float(basel_corporate_correlation([p])[0])
        if asset_correlation is None
        else _number("asset_correlation", asset_correlation, 0.0, 1.0, open_high=True)
    )
    return float(g * (conditional_pd([p], rho, confidence)[0] - p))
