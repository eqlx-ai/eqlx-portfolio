"""A portfolio run: loan file plus settings in, loss distribution and risk shares out.

Every figure in the result is produced here from the inputs; nothing is estimated from data.
The result is a plain dictionary so that it can be printed, saved or handed to an agent as is.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import config, crplus, data, factors, pricing, ratings, simulation, vasicek
from ._checks import Array
from ._version import __version__
from .errors import InputError


@dataclass(frozen=True)
class _Level:
    confidence: float
    value_at_risk: float
    var_low: float | None
    var_high: float | None
    expected_shortfall: float
    contributions: Array


@dataclass(frozen=True)
class _Run:
    expected_loss: float
    sd: float
    levels: list[_Level]
    steps: list[str]
    extra: dict[str, Any]
    without: list[tuple[float, float, float]] | None = None  # (mean, var, es) per level, flagged borrowers left out


def _factor_model(b: data.Borrowers, s: config.Settings) -> tuple[factors.FactorModel, Array | None, str]:
    """The factor model, the per-borrower one-factor correlation if there is one, and a description."""
    c = s.correlation
    n = len(b)
    if c.model == "single":
        assert c.rho is not None
        return factors.single_factor(c.rho, n), np.full(n, c.rho), f"one common factor, asset correlation {c.rho:g}"
    if c.model == "basel_corporate":
        rho = vasicek.basel_corporate_correlation(b.pd)
        text = f"one common factor, Basel corporate correlation by pd ({rho.min():.3f} to {rho.max():.3f})"
        return factors.single_factor(rho, n), rho, text
    within = config.sector_within(s, b.sector_names)
    between = config.sector_between(s, b.sector_names)
    text = (
        f"{len(b.sector_names)} sector factor(s); within-sector asset correlation {within.min():g} to {within.max():g}"
    )
    return factors.sector_factors(b.sector, within, between), None, text


def _simulate(b: data.Borrowers, s: config.Settings, exclude: Any = None, side_factors: int = 0) -> _Run:
    model, rho, text = _factor_model(b, s)
    n = len(b)
    if side_factors and np.any(model.loadings[~exclude][:, model.loadings.shape[1] - side_factors :] != 0.0):
        side_factors = 0  # perfectly correlated sectors: the new sector's factor cannot be separated
    outcomes = np.column_stack((np.zeros(n), b.amount))
    sim = simulation.simulate(
        model,
        simulation.default_thresholds(b.pd),
        outcomes,
        scenarios=s.scenarios,
        seed=s.seed,
        tail_levels=list(s.confidence),
        exclude=exclude,
        side_factors=side_factors,
        default_loss_rate_sd=b.lgd_sd,
        default_loss_rate_mean=b.lgd if b.lgd_sd is not None else None,
    )
    qs = simulation.quantiles(sim.totals, list(s.confidence))
    levels = [
        _Level(q.confidence, q.value, q.ci_low, q.ci_high, q.expected_shortfall, t.contributions)
        for q, t in zip(qs, sim.tails, strict=True)
    ]
    mean = float(sim.totals.mean())
    sd = float(sim.totals.std())
    exact_el = float(b.amount @ b.pd)
    extra: dict[str, Any] = {
        "expected_loss_exact": exact_el,
        "expected_loss_simulated": mean,
        "expected_loss_standard_error": sd / float(np.sqrt(s.scenarios)),
    }
    steps = [
        f"Grouped the loans into {n} borrower(s); a borrower's loans default together.",
        f"Correlation: {text}.",
        f"Drew {s.scenarios} scenarios with seed {s.seed}; in each, a borrower defaults when its asset return "
        "falls below the threshold set by its pd.",
    ]
    if b.lgd_sd is not None:
        steps.append("Loss given default was drawn from a beta distribution with the given mean and lgd_sd.")
    if n <= simulation.EXACT_SD_MAX_BORROWERS:
        amount_sd = None if b.lgd_sd is None else b.exposure * b.lgd_sd
        _, exact_sd = simulation.exact_loss_moments(b.amount, b.pd, model, amount_sd)
        extra["loss_sd_exact"] = exact_sd
        steps.append("Checked the simulated mean and standard deviation against the exact pairwise formulas.")
    if rho is not None:
        extra["large_portfolio_benchmark"] = [
            {
                "confidence": q,
                "loss": vasicek.large_portfolio_loss(b.amount, b.pd, rho, q),
                "unexpected_loss": vasicek.large_portfolio_loss(b.amount, b.pd, rho, q) - exact_el,
            }
            for q in s.confidence
        ]
        steps.append(
            "Computed the large-portfolio benchmark: the loss a perfectly fine-grained portfolio with the same "
            "pd, lgd and correlation would show (the formula behind the Basel risk weights)."
        )
    for q in qs:
        beyond = s.scenarios - simulation.order_index(s.scenarios, q.confidence)
        if q.ci_high is None:
            steps.append(
                f"Warning: {s.scenarios} scenarios leave only {beyond} outcome(s) at or beyond the "
                f"{q.confidence * 100:g}% level; the value-at-risk there has no upper bound. Raise scenarios."
            )
        elif beyond < 100:
            steps.append(
                f"Note: the {q.confidence * 100:g}% level rests on {beyond} outcomes; expect a wide band and noisy "
                "risk contributions. Raise scenarios for a firmer figure."
            )
    without = None
    if sim.totals_excluding is not None:
        w = sim.totals_excluding
        without = [
            (float(w.mean()), q.value, q.expected_shortfall) for q in simulation.quantiles(w, list(s.confidence))
        ]
    return _Run(exact_el, sd, levels, steps, extra, without)


def _crp_inputs(b: data.Borrowers, s: config.Settings) -> tuple[Array, Array, list[str], str]:
    c = s.correlation
    n = len(b)
    if c.model == "single":
        names = ["all"]
        weights = np.ones((n, 1))
        groups = [np.arange(n)]
        rhos = [float(c.rho or 0.0)]
    else:
        names = list(b.sector_names)
        weights = np.zeros((n, len(names)))
        weights[np.arange(n), b.sector] = 1.0
        groups = [np.nonzero(b.sector == k)[0] for k in range(len(names))]
        rhos = [float(x) for x in config.sector_within(s, b.sector_names)]
    unknown = sorted(set(s.sector_sigma) - set(names))
    if unknown:
        raise InputError(f"creditriskplus.sector_sigma names unknown sector(s): {', '.join(unknown)}")
    sigma = []
    derived = []
    for name, idx, rho in zip(names, groups, rhos, strict=True):
        if name in s.sector_sigma:
            sigma.append(s.sector_sigma[name])
        elif s.default_sigma is not None:
            sigma.append(s.default_sigma)
        else:
            avg_pd = float(b.amount[idx] @ b.pd[idx] / b.amount[idx].sum())
            sigma.append(crplus.sigma_from_asset_correlation(avg_pd, rho) if rho > 0.0 and avg_pd > 0.0 else 0.0)
            derived.append(name)
    text = "sector default-rate volatility: " + ", ".join(f"{nm} {sg:.3f}" for nm, sg in zip(names, sigma, strict=True))
    if derived:
        text += f" (derived from the asset correlation at the sector's average pd for: {', '.join(derived)})"
    return weights, np.array(sigma), names, text


def _creditriskplus(b: data.Borrowers, s: config.Settings) -> _Run:
    if b.lgd_sd is not None:
        raise InputError("CreditRisk+ uses a fixed loss given default: remove lgd_sd or use simulation")
    if np.any(b.amount <= 0.0):
        raise InputError("CreditRisk+ needs a positive loss given default for every borrower")
    weights, sigma, _, text = _crp_inputs(b, s)
    res = crplus.creditriskplus(b.amount, b.pd, weights, sigma, loss_unit=s.loss_unit, confidence=list(s.confidence))
    levels = [
        _Level(t.confidence, t.value_at_risk, None, None, t.expected_shortfall, t.contributions) for t in res.tails
    ]
    steps = [
        f"Grouped the loans into {len(b)} borrower(s); a borrower's loans default together.",
        f"Rounded each borrower's loss given default to whole multiples of {res.loss_unit:,.2f} and adjusted its "
        "pd so that its expected loss is unchanged.",
        f"CreditRisk+ with independent sectors; {text}.",
        f"Expanded the loss distribution to {res.probabilities.size - 1} loss units; probability left beyond that: "
        f"{res.tail_mass:.1e}.",
    ]
    if abs(res.sd / res.sd_before_rounding - 1.0) > 2 * crplus.ROUNDING_TOLERANCE:
        steps.append(
            f"Warning: rounding to loss units moved the standard deviation by "
            f"{res.sd / res.sd_before_rounding - 1.0:+.1%}; set a smaller creditriskplus.loss_unit."
        )
    extra = {
        "expected_loss_exact": float(b.amount @ b.pd),
        "loss_sd_before_rounding": res.sd_before_rounding,
        "loss_unit": res.loss_unit,
        "loss_units_computed": int(res.probabilities.size - 1),
        "probability_beyond_last_unit": res.tail_mass,
        "probability_of_no_loss": float(res.probabilities[0]),
    }
    return _Run(res.expected_loss, res.sd, levels, steps, extra)


def _shares(values: Array) -> Array:
    total = float(values.sum())
    out: Array = values / total if total > 0.0 else np.zeros_like(values)
    return out


def _concentration(b: data.Borrowers, contributions: Array) -> dict[str, Any]:
    share = _shares(b.exposure)
    order = np.sort(share)[::-1]
    hhi = float(share @ share)
    risk = _shares(contributions)
    by_sector: list[dict[str, Any]] = []
    for k, name in enumerate(b.sector_names):
        m = b.sector == k
        by_sector.append(
            {
                "sector": name,
                "borrowers": int(m.sum()),
                "exposure": float(b.exposure[m].sum()),
                "exposure_share": float(share[m].sum()),
                "expected_loss": float(b.amount[m] @ b.pd[m]),
                "risk_contribution": float(contributions[m].sum()),
                "risk_share": float(risk[m].sum()),
            }
        )
    by_sector.sort(key=lambda r: (-float(r["risk_contribution"]), str(r["sector"])))
    return {
        "herfindahl_index": hhi,
        "effective_number_of_borrowers": 1.0 / hhi,
        "largest_borrower_share": float(order[0]),
        "ten_largest_share": float(order[:10].sum()),
        "by_sector": by_sector,
    }


def _borrower_rows(b: data.Borrowers, contributions: Array, top: int, only: Any = None) -> list[dict[str, Any]]:
    """Borrowers by risk contribution, largest first. ``only`` keeps the flagged borrowers; shares stay
    shares of the whole portfolio."""
    risk = _shares(contributions)
    order = np.lexsort((np.arange(len(b)), -contributions))
    if only is not None:
        order = order[only[order]]
    return [
        {
            "borrower_id": b.borrower_id[i],
            "sector": b.sector_names[int(b.sector[i])],
            "loans": int(b.loans[i]),
            "exposure": float(b.exposure[i]),
            "pd": float(b.pd[i]),
            "lgd": float(b.lgd[i]),
            "expected_loss": float(b.amount[i] * b.pd[i]),
            "risk_contribution": float(contributions[i]),
            "risk_share": float(risk[i]),
            "exposure_share": float(b.exposure[i] / b.exposure.sum()),
        }
        for i in order[:top]
    ]


def _level_rows(run: _Run) -> list[dict[str, Any]]:
    rows = []
    for lv in run.levels:
        row: dict[str, Any] = {
            "confidence": lv.confidence,
            "value_at_risk": lv.value_at_risk,
            "expected_shortfall": lv.expected_shortfall,
            "economic_capital": lv.value_at_risk - run.expected_loss,
        }
        if lv.var_low is not None:
            row["value_at_risk_95_band"] = [lv.var_low, lv.var_high]
        rows.append(row)
    return rows


CONVENTIONS = [
    "Horizon: one period, the period the pd values refer to (normally one year).",
    "Default mode: a loan loses exposure x lgd if its borrower defaults and nothing otherwise; rating changes "
    "short of default are not valued in a portfolio run.",
    "Value-at-risk at a confidence level is the loss not exceeded in that share of outcomes; expected shortfall "
    "is the average loss in the outcomes at or beyond it; economic capital is value-at-risk minus expected loss.",
    "Risk contribution: a borrower's average loss in the outcomes that make up the expected shortfall at the "
    "highest confidence level. Contributions add up to that expected shortfall.",
    "pd, lgd, exposure and correlations are inputs. The tool does not estimate them.",
]


def _result(loans: data.Loans, b: data.Borrowers, s: config.Settings, run: _Run) -> dict[str, Any]:
    top_level = run.levels[-1]
    return {
        "tool": {"name": "eqlx-portfolio", "version": __version__, "numpy": np.__version__},
        "method": s.method,
        "settings": s.to_dict(),
        "portfolio": {
            "loans": len(loans),
            "borrowers": len(b),
            "sectors": len(b.sector_names),
            "exposure": float(b.exposure.sum()),
            "loss_if_all_default": float(b.amount.sum()),
        },
        "expected_loss": run.expected_loss,
        "loss_sd": run.sd,
        "levels": _level_rows(run),
        "checks": run.extra,
        "concentration": _concentration(b, top_level.contributions),
        "contributions_at_confidence": top_level.confidence,
        "top_contributors": _borrower_rows(b, top_level.contributions, s.top),
        "steps": run.steps,
        "conventions": CONVENTIONS,
    }


def _compute(b: data.Borrowers, s: config.Settings, exclude: Any = None, side_factors: int = 0) -> _Run:
    if s.method == "creditriskplus":
        return _creditriskplus(b, s)
    return _simulate(b, s, exclude, side_factors)


def _for_sectors(s: config.Settings, sector_names: tuple[str, ...]) -> config.Settings:
    """The same settings without entries for sectors that are not in this book."""
    c = s.correlation
    keep = set(sector_names)
    corr = dataclasses.replace(
        c,
        within={k: v for k, v in c.within.items() if k in keep},
        between={k: v for k, v in c.between.items() if k[0] in keep and k[1] in keep},
    )
    sigma = {k: v for k, v in s.sector_sigma.items() if k in keep or c.model == "single"}
    return dataclasses.replace(s, correlation=corr, sector_sigma=sigma)


def run(loans: data.Loans, settings: config.Settings) -> dict[str, Any]:
    """Loss distribution, capital and risk shares of a loan portfolio."""
    b = data.group(loans)
    return _result(loans, b, settings, _compute(b, settings))


def contributions(loans: data.Loans, settings: config.Settings) -> list[dict[str, Any]]:
    """Every borrower's risk contribution, largest first."""
    b = data.group(loans)
    res = _compute(b, settings)
    return _borrower_rows(b, res.levels[-1].contributions, len(b))


def _merge(loans: data.Loans, new: data.Loans) -> data.Loans:
    clash = sorted(set(loans.loan_id) & set(new.loan_id))
    if clash:
        raise InputError(f"new loan_id already in the portfolio: {clash[0]}")
    sd = None
    if loans.lgd_sd is not None or new.lgd_sd is not None:
        sd = np.concatenate([x.lgd_sd if x.lgd_sd is not None else np.zeros(len(x)) for x in (loans, new)])
    return data.Loans(
        loans.loan_id + new.loan_id,
        loans.borrower_id + new.borrower_id,
        loans.sector + new.sector,
        np.concatenate((loans.exposure, new.exposure)),
        np.concatenate((loans.pd, new.pd)),
        np.concatenate((loans.lgd, new.lgd)),
        sd,
    )


def what_if(loans: data.Loans, new_loans: data.Loans, settings: config.Settings) -> dict[str, Any]:
    """What adding ``new_loans`` does to the portfolio's loss, capital and concentration.

    With simulation, the portfolio with and without the new loans is valued on the same
    scenarios, so the difference is not blurred by random noise. New loans must belong to
    borrowers that are not yet in the portfolio.
    """
    known = set(loans.borrower_id)
    again = sorted({x for x in new_loans.borrower_id if x in known})
    if again:
        raise InputError(f"borrower {again[0]!r} is already in the portfolio; what-if supports new borrowers only")
    merged = _merge(loans, new_loans)
    old = data.group(loans)
    # Sectors of the existing book keep their numbers, new sectors follow: the existing borrowers
    # then see exactly the factors and random numbers of a run without the new loans.
    b = data.group(merged, sector_order=old.sector_names)
    new_names = set(new_loans.borrower_id)
    flag = np.array([name in new_names for name in b.borrower_id])
    before_settings = _for_sectors(settings, old.sector_names)
    if settings.method == "simulation":
        side = len(b.sector_names) - len(old.sector_names) if settings.correlation.model == "sector" else 0
        after = _compute(b, settings, flag, side)
        assert after.without is not None
        before_el = float(old.amount @ old.pd)
        before_levels = [(v, es) for _, v, es in after.without]
    else:
        after = _compute(b, settings)
        before = _compute(old, before_settings)
        before_el = before.expected_loss
        before_levels = [(lv.value_at_risk, lv.expected_shortfall) for lv in before.levels]
    rows = []
    for lv, (var0, es0) in zip(after.levels, before_levels, strict=True):
        rows.append(
            {
                "confidence": lv.confidence,
                "value_at_risk_before": var0,
                "value_at_risk_after": lv.value_at_risk,
                "economic_capital_before": var0 - before_el,
                "economic_capital_after": lv.value_at_risk - after.expected_loss,
                "economic_capital_change": (lv.value_at_risk - after.expected_loss) - (var0 - before_el),
                "expected_shortfall_before": es0,
                "expected_shortfall_after": lv.expected_shortfall,
                "expected_shortfall_change": lv.expected_shortfall - es0,
                "risk_contribution_of_new_loans": float(lv.contributions[flag].sum()),
            }
        )
    hhi_before = float(_shares(old.exposure) @ _shares(old.exposure))
    hhi_after = float(_shares(b.exposure) @ _shares(b.exposure))
    new_exposure = float(b.exposure[flag].sum())
    top = after.levels[-1]
    return {
        "tool": {"name": "eqlx-portfolio", "version": __version__, "numpy": np.__version__},
        "method": settings.method,
        "settings": settings.to_dict(),
        "new_loans": {
            "loans": len(new_loans),
            "borrowers": int(flag.sum()),
            "exposure": new_exposure,
            "expected_loss": float(b.amount[flag] @ b.pd[flag]),
        },
        "expected_loss_before": before_el,
        "expected_loss_after": after.expected_loss,
        "levels": rows,
        "capital_per_unit_of_new_exposure": rows[-1]["expected_shortfall_change"] / new_exposure,
        "herfindahl_index_before": hhi_before,
        "herfindahl_index_after": hhi_after,
        "new_borrowers": _borrower_rows(b, top.contributions, int(flag.sum()), only=flag),
        "conventions": [
            *CONVENTIONS,
            "capital_per_unit_of_new_exposure is the change in expected shortfall at the highest confidence level "
            "divided by the new exposure.",
        ],
    }


_TERMS = {
    "years",
    "funding",
    "target_return",
    "operating_cost",
    "capital_return",
    "tax_rate",
    "payments_per_year",
    "repayment",
    "rate",
    "risk_weight",
    "capital_ratio",
    "pd_cumulative",
}


def price_new_loan(loans: data.Loans, new_loan: data.Loans, settings: config.Settings, terms: Any) -> dict[str, Any]:
    """What one new loan has to earn, with the capital it ties up measured three ways.

    ``terms`` holds the loan's life and the bank's costs: ``years``, ``funding`` (a rate or a list of
    [years, rate] points), ``target_return``, and optionally ``operating_cost``, ``capital_return``,
    ``tax_rate``, ``payments_per_year``, ``repayment``, the offered ``rate``, ``pd_cumulative`` and,
    for the regulatory view, ``risk_weight`` with ``capital_ratio``.

    Capital per unit lent is taken from (1) this portfolio: the rise in expected shortfall at the
    highest confidence level beyond the loan's own expected loss, (2) the one-factor formula
    behind the Basel risk weights, and (3) a fixed risk weight if one is given. Each gives its own
    hurdle rate; the gap between them is what the book's concentration is worth.
    """
    if not isinstance(terms, dict):
        raise InputError("terms must be an object")
    extra = sorted(set(terms) - _TERMS)
    if extra:
        raise InputError(f"unknown term(s): {', '.join(extra)}")
    missing = [k for k in ("years", "funding", "target_return") if k not in terms]
    if missing:
        raise InputError(f"terms need: {', '.join(missing)}")
    if ("risk_weight" in terms) != ("capital_ratio" in terms):
        raise InputError("risk_weight and capital_ratio go together")
    if len(new_loan) != 1:
        raise InputError("price one new loan at a time")
    w = what_if(loans, new_loan, settings)
    exposure = float(new_loan.exposure[0])
    pd, lgd = float(new_loan.pd[0]), float(new_loan.lgd[0])
    if pd <= 0.0 or lgd <= 0.0:
        raise InputError("the new loan needs a positive pd and lgd to be priced")
    top = w["levels"][-1]
    own_el = w["new_loans"]["expected_loss"]
    portfolio_capital = (top["expected_shortfall_change"] - own_el) / exposure
    common = {
        "notional": exposure,
        "years": terms["years"],
        "pd": terms.get("pd_cumulative", pd),
        "lgd": lgd,
        "target_return": terms["target_return"],
        "funding": terms["funding"],
        "operating_cost": terms.get("operating_cost", 0.0),
        "capital_return": terms.get("capital_return", 0.0),
        "tax_rate": terms.get("tax_rate", 0.0),
        "payments_per_year": terms.get("payments_per_year", 1),
        "repayment": terms.get("repayment", "bullet"),
        "rate": terms.get("rate"),
    }
    bases: list[tuple[str, float, str]] = [
        (
            "portfolio",
            portfolio_capital,
            f"rise in this book's expected shortfall at {top['confidence'] * 100:g}% beyond the loan's expected loss",
        ),
        (
            "basel_formula",
            pricing.capital_from_basel_formula(pd, lgd),
            "one-factor formula behind the Basel risk weights at 99.9%, no maturity adjustment",
        ),
    ]
    if "risk_weight" in terms:
        bases.append(
            (
                "risk_weight",
                pricing.capital_from_risk_weight(terms["risk_weight"], terms["capital_ratio"]),
                "fixed risk weight times the capital ratio held",
            )
        )
    views = []
    for name, capital, meaning in bases:
        if capital <= 0.0:
            views.append(
                {
                    "capital_basis": name,
                    "meaning": meaning,
                    "capital_per_unit_lent": capital,
                    "not_priced": "this basis shows no capital need for the loan; with simulation, the loan may "
                    "be too small to measure: raise scenarios or use another basis",
                }
            )
            continue
        views.append(
            {"capital_basis": name, "meaning": meaning, **pricing.price_loan(capital=capital, **common).to_dict()}
        )
    return {
        "tool": {"name": "eqlx-portfolio", "version": __version__, "numpy": np.__version__},
        "loan": {
            "loan_id": new_loan.loan_id[0],
            "sector": new_loan.sector[0],
            "exposure": exposure,
            "pd": pd,
            "lgd": lgd,
            "expected_loss_first_year": own_el,
        },
        "terms": dict(terms),
        "portfolio_effect": {
            "confidence": top["confidence"],
            "expected_shortfall_change": top["expected_shortfall_change"],
            "economic_capital_change": top["economic_capital_change"],
        },
        "views": views,
        "settings": settings.to_dict(),
        "conventions": [
            "All rates are yearly decimals. The hurdle rate is the interest rate at which RAROC equals the target "
            "return: funding cost + expected loss margin + operating cost margin + capital margin.",
            "RAROC = (1 - tax rate) x (rate - funding cost - expected loss margin - operating cost margin) "
            "/ capital per unit lent + what the capital earns where it is invested.",
            "Capital is measured over one year and held at the same share of the amount lent for the life of the loan.",
            "Costs, target return, pd and lgd are inputs. The hurdle rate is a break-even point, not a "
            "recommended price.",
        ],
    }


def pair(pd_a: float, pd_b: float, asset_correlation: float) -> dict[str, Any]:
    """Joint default probability of two borrowers; the default correlation is None when a pd is 0 or 1."""
    joint = float(ratings.joint_default_probability(pd_a, pd_b, asset_correlation)[0])
    defined = 0.0 < pd_a < 1.0 and 0.0 < pd_b < 1.0
    return {
        "pd_a": pd_a,
        "pd_b": pd_b,
        "asset_correlation": asset_correlation,
        "joint_default_probability": joint,
        "if_independent": pd_a * pd_b,
        "default_correlation": float(ratings.default_correlation(pd_a, pd_b, asset_correlation)[0])
        if defined
        else None,
    }


def describe(loans: data.Loans) -> dict[str, Any]:
    """What a loan file contains, before any correlation assumption is applied."""
    b = data.group(loans)
    conc = _concentration(b, b.amount * b.pd)
    for row in conc["by_sector"]:
        row["expected_loss_share"] = row.pop("risk_share")
        del row["risk_contribution"]
    conc["by_sector"].sort(key=lambda r: (-float(r["exposure"]), str(r["sector"])))
    order = np.lexsort((np.arange(len(b)), -b.exposure))[:10]
    return {
        "loans": len(loans),
        "borrowers": len(b),
        "sectors": list(b.sector_names),
        "exposure": float(b.exposure.sum()),
        "expected_loss": float(b.amount @ b.pd),
        "loss_if_all_default": float(b.amount.sum()),
        "pd_exposure_weighted": float(b.exposure @ b.pd / b.exposure.sum()),
        "lgd_exposure_weighted": float(b.amount.sum() / b.exposure.sum()),
        "has_lgd_sd": b.lgd_sd is not None,
        "concentration": conc,
        "largest_borrowers": [
            {
                "borrower_id": b.borrower_id[i],
                "sector": b.sector_names[int(b.sector[i])],
                "exposure": float(b.exposure[i]),
                "exposure_share": float(b.exposure[i] / b.exposure.sum()),
                "pd": float(b.pd[i]),
            }
            for i in order
        ],
    }
