"""The one-factor large-portfolio benchmark (Vasicek 1987, 2002; Gordy 2003).

In a portfolio so finely spread that no single borrower matters, the loss in a bad year is each
borrower's loss given default times its default probability *conditional on the bad state of
the common factor*. This is the model behind the Basel internal ratings-based risk weights, and
the limit a simulation must approach as the portfolio gets finer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri

from ._checks import Array, as_1d, broadcast, in_unit_interval, non_negative
from .errors import InputError


def conditional_pd(pd: ArrayLike, rho: ArrayLike, confidence: float) -> Array:
    """Default probability in the year that is worse than ``confidence`` of all years."""
    if not 0.0 < confidence < 1.0:
        raise InputError("confidence must lie strictly between 0 and 1")
    p = in_unit_interval("pd", as_1d("pd", pd))
    r = in_unit_interval("rho", broadcast("rho", rho, p.size))
    if np.any(r >= 1.0):
        raise InputError("rho must be below 1")
    with np.errstate(divide="ignore"):
        out: Array = ndtr((ndtri(p) + np.sqrt(r) * ndtri(confidence)) / np.sqrt(1.0 - r))
    return out


def large_portfolio_loss(amount: ArrayLike, pd: ArrayLike, rho: ArrayLike, confidence: float) -> float:
    """Loss at ``confidence`` of an infinitely fine portfolio. ``amount`` is exposure times loss given default."""
    p = as_1d("pd", pd)
    a = non_negative("amount", broadcast("amount", amount, p.size))
    return float(a @ conditional_pd(p, rho, confidence))


def loss_rate_cdf(x: ArrayLike, pd: float, rho: float) -> Array:
    """Distribution of the default rate of a large pool of identical borrowers (Vasicek 2002)."""
    if not 0.0 < pd < 1.0 or not 0.0 < rho < 1.0:
        raise InputError("pd and rho must lie strictly between 0 and 1")
    v = in_unit_interval("x", as_1d("x", x))
    with np.errstate(divide="ignore"):
        out: Array = ndtr((np.sqrt(1.0 - rho) * ndtri(v) - ndtri(pd)) / np.sqrt(rho))
    return out


def basel_corporate_correlation(pd: ArrayLike) -> Array:
    """Asset correlation of the Basel corporate risk-weight function: 24% for the safest names, 12% for the riskiest."""
    p = in_unit_interval("pd", as_1d("pd", pd))
    w = (1.0 - np.exp(-50.0 * p)) / (1.0 - np.exp(-50.0))
    out: Array = 0.12 * w + 0.24 * (1.0 - w)
    return out
