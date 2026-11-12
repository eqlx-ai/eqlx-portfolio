"""Value of one exposure at the risk horizon, state by state, and the spread of that value.

Follows chapters 1, 2 and 9 of the CreditMetrics Technical Document (1997): the exposure is
revalued in every possible year-end rating with that rating's forward zero curve, takes its
recovery value in default, and the probabilities of the states give the mean and the standard
deviation of value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._checks import Array, as_1d, as_array, non_negative, probability_row
from .errors import InputError
from .ratings import joint_migration


def bond_values_at_horizon(
    face: float,
    coupon_rate: float,
    years_to_maturity: int,
    forward_curves: ArrayLike,
    recovery_rate: float,
) -> Array:
    """Value one year from now of an annual-coupon bond or bullet loan, in each year-end state.

    ``forward_curves`` has one row per non-default rating (best first); column ``t - 1`` is the
    one-year-forward zero rate for a cash flow ``t`` years after the horizon, as a decimal.
    The value in a rating is the coupon paid at the horizon plus the remaining cash flows
    discounted on that rating's curve. In default it is ``face * recovery_rate``. The result has
    one entry per rating followed by default.
    """
    if years_to_maturity < 1 or int(years_to_maturity) != years_to_maturity:
        raise InputError("years_to_maturity must be a whole number of years, at least 1")
    if not 0.0 <= recovery_rate <= 1.0:
        raise InputError("recovery_rate must lie in [0, 1]")
    if face < 0.0 or coupon_rate < 0.0:
        raise InputError("face and coupon_rate must not be negative")
    curves = as_array("forward_curves", forward_curves, 2)
    remaining = int(years_to_maturity) - 1
    if curves.shape[1] < remaining:
        raise InputError(f"forward_curves cover {curves.shape[1]} years, the exposure needs {remaining}")
    if np.any(curves <= -1.0):
        raise InputError("forward rates must be above -100%")
    coupon = face * coupon_rate
    flows = np.full(remaining, coupon)
    if remaining:
        flows[-1] += face
        t = np.arange(1, remaining + 1)
        pv = (flows / (1.0 + curves[:, :remaining]) ** t).sum(axis=1)
        alive = coupon + pv
    else:
        alive = np.full(curves.shape[0], coupon + face)
    return np.concatenate((alive, [face * recovery_rate]))


@dataclass(frozen=True)
class ValueStats:
    mean: float
    variance: float
    sd: float


def value_stats(probs: ArrayLike, values: ArrayLike, value_sds: ArrayLike | None = None) -> ValueStats:
    """Mean and standard deviation of one exposure's value across its year-end states.

    ``value_sds`` optionally gives the uncertainty of the value inside each state (for example
    the standard deviation of recovery in default); its variance is added, weighted by the
    state's probability (Technical Document eq. 2.3).
    """
    p = probability_row("probs", probs)
    v = as_1d("values", values)
    if v.size != p.size:
        raise InputError("values needs one entry per state")
    mean = float(p @ v)
    variance = float(p @ (v - mean) ** 2)
    if value_sds is not None:
        s = non_negative("value_sds", as_1d("value_sds", value_sds))
        if s.size != p.size:
            raise InputError("value_sds needs one entry per state")
        variance += float(p @ (s * s))
    return ValueStats(mean, variance, float(np.sqrt(variance)))


def percentile_of_states(probs: ArrayLike, values: ArrayLike, level: float) -> float:
    """Lowest value ``v`` such that the probability of a value of ``v`` or less is at least ``level``."""
    p = as_1d("probs", np.ravel(np.asarray(probs, dtype=np.float64)))
    v = as_1d("values", np.ravel(np.asarray(values, dtype=np.float64)))
    if p.size != v.size:
        raise InputError("probs and values must have the same number of entries")
    if not 0.0 < level < 1.0:
        raise InputError("level must lie strictly between 0 and 1")
    order = np.argsort(v, kind="stable")
    cum = np.cumsum(p[order])
    idx = int(np.searchsorted(cum, level * cum[-1] - 1e-12))
    return float(v[order][min(idx, v.size - 1)])


@dataclass(frozen=True)
class PortfolioValueStats:
    mean: float
    sd: float
    asset_means: Array
    asset_sds: Array
    covariance: Array
    marginal_sds: Array


def portfolio_value_stats(
    probs: Sequence[ArrayLike],
    values: Sequence[ArrayLike],
    correlations: ArrayLike,
    value_sds: Sequence[ArrayLike] | None = None,
) -> PortfolioValueStats:
    """Exact mean and standard deviation of portfolio value, without simulation.

    Works pair by pair (Technical Document chapter 9 and appendix A): the covariance of two
    exposures comes from their joint migration probabilities. ``probs[i]`` and ``values[i]``
    describe exposure ``i`` state by state; ``correlations`` is the asset correlation matrix.
    Each exposure must belong to a different borrower. ``marginal_sds[i]`` is the fall in
    portfolio standard deviation if exposure ``i`` is removed.
    """
    n = len(probs)
    if n == 0 or len(values) != n:
        raise InputError("probs and values need one entry per exposure")
    corr = as_array("correlations", correlations, 2)
    if corr.shape != (n, n) or not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1.0):
        raise InputError("correlations must be a symmetric n x n matrix with ones on the diagonal")
    p = [probability_row(f"probs[{i}]", row) for i, row in enumerate(probs)]
    v = [as_1d(f"values[{i}]", row) for i, row in enumerate(values)]
    stats = [value_stats(p[i], v[i], None if value_sds is None else value_sds[i]) for i in range(n)]
    means = np.array([s.mean for s in stats])
    cov = np.diag([s.variance for s in stats])
    for i in range(n):
        for j in range(i + 1, n):
            joint = joint_migration(p[i], p[j], float(corr[i, j]))
            cov[i, j] = cov[j, i] = float(v[i] @ joint @ v[j]) - means[i] * means[j]
    total_var = float(cov.sum())
    sd = float(np.sqrt(max(total_var, 0.0)))
    marginal = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        marginal[i] = sd - float(np.sqrt(max(cov[np.ix_(keep, keep)].sum(), 0.0)))
    return PortfolioValueStats(float(means.sum()), sd, means, np.sqrt(np.diag(cov)), cov, marginal)
