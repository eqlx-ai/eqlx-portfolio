"""CreditRisk+: the portfolio loss distribution in closed form, without simulation.

The model of Credit Suisse Financial Products (1997). Losses are counted in whole multiples of
a loss unit. Each borrower defaults with a small probability that is scaled up or down by the
state of its sectors; a sector's state is a gamma-distributed factor with mean one and a given
relative standard deviation. Given the sector states, defaults are independent and treated as
Poisson events. The distribution of the total loss then has a closed-form generating function,
which is expanded term by term.

The expansion follows Haaf, Reiss and Schoenmakers (2004): it works with the logarithm of the
generating function, whose series has only non-negative terms, so there is no cancellation and
the result stays accurate in portfolios where the original Panjer recursion fails.

Risk contributions follow Haaf and Tasche (2002): a borrower's share of the expected
shortfall is its expected loss in the worst outcomes, computed from the same expansion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._checks import Array, as_array, broadcast, confidence_levels, in_unit_interval, non_negative
from .errors import InputError
from .ratings import joint_default_probability

MAX_UNITS = 60_000  # the expansion costs time in proportion to the square of this
DEFAULT_UNITS = 20_000
_RESCALE_AT = 1e200


@dataclass(frozen=True)
class TailShare:
    confidence: float
    value_at_risk: float
    expected_shortfall: float
    contributions: Array  # per borrower, in currency; sums to expected_shortfall


@dataclass(frozen=True)
class CreditRiskPlus:
    loss_unit: float
    probabilities: Array  # probabilities[n] = P(loss = n * loss_unit)
    expected_loss: float
    sd: float
    tail_mass: float  # probability beyond the last term kept
    units: Array  # each borrower's loss given default in loss units
    adjusted_pd: Array  # default probability after rounding, chosen to keep the expected loss
    tails: tuple[TailShare, ...]

    def cdf(self) -> Array:
        out: Array = np.cumsum(self.probabilities)
        return out


def _log_series(a: Array, n_max: int) -> Array:
    """Coefficients 1..n_max of ``-ln(1 - A(z))`` where ``A`` has non-negative coefficients ``a`` and a[0] = 0."""
    b = np.zeros(n_max + 1)
    top = a.size - 1
    idx = np.arange(n_max + 1, dtype=np.float64)
    for n in range(1, n_max + 1):
        lo = max(1, n - top)
        acc = a[n] if n <= top else 0.0
        if lo < n:
            # sum_{j=lo}^{n-1} j * b[j] * a[n-j]
            acc += float((idx[lo:n] * b[lo:n]) @ a[n - lo : 0 : -1]) / n
        b[n] = acc
    return b


def _exp_series(e: Array, log_constant: float, n_max: int) -> tuple[Array, float]:
    """Coefficients of ``exp(log_constant + sum_{n>=1} e[n] z^n)`` as ``values * exp(log_scale)``."""
    g = np.zeros(n_max + 1)
    g[0] = 1.0
    log_scale = log_constant
    je = np.arange(n_max + 1, dtype=np.float64) * e
    for n in range(1, n_max + 1):
        g[n] = float(je[1 : n + 1] @ g[n - 1 :: -1][:n]) / n
        if g[n] > _RESCALE_AT:
            g[: n + 1] /= _RESCALE_AT
            log_scale += float(np.log(_RESCALE_AT))
    return g, log_scale


def _to_probabilities(g: Array, log_scale: float) -> Array:
    with np.errstate(divide="ignore"):
        out: Array = np.exp(np.log(g) + log_scale)
    return out


def sigma_from_asset_correlation(pd: float, rho: float) -> float:
    """Relative standard deviation of the default rate that matches a one-factor asset correlation.

    In the one-factor asset value model the variance of the conditional default probability is
    the joint default probability minus the squared default probability; dividing its square
    root by the default probability gives the CreditRisk+ sector volatility with the same
    default correlation (Gordy 2000).
    """
    if not 0.0 < pd < 1.0 or not 0.0 <= rho < 1.0:
        raise InputError("pd must lie strictly between 0 and 1 and rho in [0, 1)")
    joint = float(joint_default_probability(pd, pd, rho)[0])
    return float(np.sqrt(max(joint - pd * pd, 0.0)) / pd)


def creditriskplus(
    amount: ArrayLike,
    pd: ArrayLike,
    sector_weights: ArrayLike,
    sector_sigma: ArrayLike,
    *,
    loss_unit: float | None = None,
    confidence: ArrayLike | None = None,
    tail_tolerance: float = 1e-9,
) -> CreditRiskPlus:
    """Loss distribution, value-at-risk, expected shortfall and risk contributions.

    ``amount[i]`` is borrower ``i``'s loss if it defaults (exposure times loss given default).
    ``sector_weights[i, k]`` is the share of the borrower's default risk driven by sector ``k``;
    what is left to one is borrower-specific. ``sector_sigma[k]`` is the relative standard
    deviation of sector ``k``'s default rate (0.75 means the rate typically moves by 75% of its
    mean). Sectors are independent of each other.

    Amounts are rounded to whole multiples of ``loss_unit`` (at least one) and the default
    probability is adjusted so that each borrower's expected loss is unchanged. The default
    unit is the largest amount divided by 200, or larger when that would need more than about
    20,000 units to cover the distribution.
    """
    p = in_unit_interval("pd", as_array("pd", pd, 1))
    n = p.size
    a = non_negative("amount", broadcast("amount", amount, n))
    w = as_array("sector_weights", sector_weights, 2)
    if w.shape[0] != n:
        raise InputError("sector_weights needs one row per borrower")
    if np.any(w < 0.0) or np.any(w.sum(axis=1) > 1.0 + 1e-9):
        raise InputError("sector weights must be non-negative and sum to at most one per borrower")
    sig = non_negative("sector_sigma", as_array("sector_sigma", sector_sigma, 1))
    if sig.size != w.shape[1]:
        raise InputError("sector_sigma needs one entry per sector")
    if np.any(a <= 0.0):
        raise InputError("amount must be positive for every borrower (drop borrowers with nothing at risk)")
    levels = None if confidence is None else confidence_levels(confidence)

    if loss_unit is None:
        # Fine enough to tell the large loans apart, coarse enough to keep the expansion short.
        rough_var = float(p @ (a * a)) + float((sig * sig) @ ((w * (p * a)[:, None]).sum(axis=0) ** 2))
        unit = max(float(a.max()) / 200.0, (float(p @ a) + 12.0 * float(np.sqrt(rough_var))) / DEFAULT_UNITS)
        unit = min(unit, float(a.max()))
    else:
        unit = float(loss_unit)
    if not np.isfinite(unit) or unit <= 0.0:
        raise InputError("loss_unit must be positive")
    if unit > float(a.max()):
        raise InputError("loss_unit is larger than the largest loss given default")
    nu = np.maximum(1, np.rint(a / unit)).astype(np.intp)
    p_adj = p * a / (nu * unit)
    if np.any(p_adj > 1.0):
        raise InputError("loss_unit is too large: rounding pushes a default probability above 1")
    top = int(nu.max())
    w0 = np.clip(1.0 - w.sum(axis=1), 0.0, 1.0)

    # Mean and variance are known in closed form; they size the expansion.
    mean_units = float(p_adj @ nu)
    var_units = float(p_adj @ (nu * nu).astype(np.float64))
    sector_el = (w * (p_adj * nu)[:, None]).sum(axis=0)
    var_units += float((sig * sig) @ (sector_el * sector_el))
    sd_units = float(np.sqrt(var_units))

    def poly(weights: Array) -> Array:
        out = np.zeros(top + 1)
        np.add.at(out, nu, weights * p_adj)
        return out

    a0 = poly(w0)
    active = [k for k in range(sig.size) if sig[k] > 0.0 and float((w[:, k] * p_adj).sum()) > 0.0]
    # A sector with zero volatility adds no common movement: its borrowers act as specific risk.
    for k in range(sig.size):
        if k not in active:
            a0 = a0 + poly(w[:, k])
    mu0 = float(a0.sum())
    polys = {k: poly(w[:, k]) for k in active}
    mus = {k: float(polys[k].sum()) for k in active}
    var_k = {k: float(sig[k] ** 2) for k in active}

    n_max = int(min(MAX_UNITS, np.ceil(mean_units + 12.0 * sd_units) + top))
    while True:
        e = np.zeros(n_max + 1)
        e[: top + 1] += a0
        log_const = -mu0
        logs: dict[int, Array] = {}
        for k in active:
            c = var_k[k] / (1.0 + var_k[k] * mus[k])
            logs[k] = _log_series(c * polys[k], n_max)
            e += logs[k] / var_k[k]
            log_const -= float(np.log1p(var_k[k] * mus[k])) / var_k[k]
        g, log_scale = _exp_series(e, log_const, n_max)
        probs = _to_probabilities(g, log_scale)
        tail = max(0.0, 1.0 - float(probs.sum()))
        if tail <= tail_tolerance or n_max >= MAX_UNITS:
            break
        n_max = min(MAX_UNITS, 2 * n_max)
    if tail > max(tail_tolerance, 1e-6):
        raise InputError(
            "the loss distribution does not fit in the allowed number of loss units; use a larger loss_unit"
        )

    tails: list[TailShare] = []
    if levels is not None:
        cdf = np.cumsum(probs)

        # survival[m] = P(L >= m units) for each distribution used below
        def survival(pr: Array) -> Array:
            s = np.cumsum(pr[::-1])[::-1]
            return np.minimum(s + max(0.0, 1.0 - float(pr.sum())), 1.0)

        surv = {-1: survival(probs)}
        for k in active:
            gk, lk = _exp_series(e + logs[k], log_const - float(np.log1p(var_k[k] * mus[k])), n_max)
            surv[k] = survival(_to_probabilities(gk, lk))
        for q in levels:
            v = int(np.searchsorted(cdf, float(q) - 1e-12))
            if v > n_max:
                raise InputError("confidence is beyond the computed part of the distribution")
            p_tail = float(surv[-1][v])
            shifted = np.maximum(v - nu, 0)
            inner = w0 * surv[-1][shifted]
            for k in range(sig.size):
                inner = inner + w[:, k] * surv[k if k in active else -1][shifted]
            contrib = nu * p_adj * inner / p_tail
            tails.append(TailShare(float(q), v * unit, float(contrib.sum()) * unit, contrib * unit))

    return CreditRiskPlus(
        unit, probs, mean_units * unit, sd_units * unit, tail, nu.astype(np.float64), p_adj, tuple(tails)
    )
