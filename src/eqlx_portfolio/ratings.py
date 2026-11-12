"""Credit quality changes as moves of a standard normal "asset return" across thresholds.

A borrower's rating one year from now is decided by where its asset return falls: below the
lowest threshold it defaults, above the highest it reaches the best grade. The thresholds are
set so that each band has exactly the borrower's migration probability. Two borrowers move
together through the correlation of their asset returns. This is the asset value model of the
CreditMetrics Technical Document (1997), sections 3.3 and 8.4.

Rating states are always ordered from best to worst, with default last.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from ._bvn import bvn_cdf
from ._checks import Array, as_array, correlation, in_unit_interval, probability_row
from .errors import InputError


def _cuts(probs: Array) -> Array:
    """Band edges in ascending order of asset return: edge j is below the j worst states."""
    # Rounding can push a partial sum a hair above one; the inverse normal of that is not a number.
    from_worst = np.minimum(np.cumsum(probs[::-1]), 1.0)
    # Everything from the best state that can actually occur upwards is certain: exactly one.
    from_worst[int(np.flatnonzero(probs[::-1] > 0.0)[-1]) :] = 1.0
    with np.errstate(divide="ignore"):
        inner: Array = ndtri(from_worst[:-1])
    return np.concatenate(([-np.inf], inner, [np.inf]))


def migration_thresholds(probs: ArrayLike) -> Array:
    """Asset return thresholds for one borrower.

    ``probs`` are the one-period probabilities of ending in each state, best first and default
    last. The result has one entry per state except the best: entry ``s - 1`` is the threshold
    below which the borrower ends in state ``s`` or worse, so the last entry is the default
    threshold. A state with zero probability shares its threshold with its neighbour.
    """
    p = probability_row("probs", probs)
    if p.size < 2:
        raise InputError("probs needs at least two states (the last one is default)")
    out: Array = _cuts(p)[1:-1][::-1].copy()
    return out


def joint_migration(probs_a: ArrayLike, probs_b: ArrayLike, rho: float) -> Array:
    """Joint probabilities of the year-end states of two borrowers.

    Entry ``[s, t]`` is the probability that borrower A ends in state ``s`` and borrower B in
    state ``t`` (best first, default last) when their asset returns have correlation ``rho``.
    Rows sum to A's probabilities and columns to B's.
    """
    pa = probability_row("probs_a", probs_a)
    pb = probability_row("probs_b", probs_b)
    r = correlation("rho", rho)
    ca = _cuts(pa)
    cb = _cuts(pb)
    grid = bvn_cdf(ca[:, None], cb[None, :], r)
    cells = np.diff(np.diff(grid, axis=0), axis=1)
    cells = np.maximum(cells, 0.0)
    out: Array = cells[::-1, ::-1].copy()
    return out


def joint_default_probability(pd_a: ArrayLike, pd_b: ArrayLike, rho: ArrayLike) -> Array:
    """Probability that both borrowers default, given their default probabilities and asset correlation."""
    a = in_unit_interval("pd_a", as_array("pd_a", np.atleast_1d(pd_a), 1))
    b = in_unit_interval("pd_b", as_array("pd_b", np.atleast_1d(pd_b), 1))
    r = as_array("rho", np.atleast_1d(rho), 1)
    if np.any(np.abs(r) > 1.0):
        raise InputError("rho must lie in [-1, 1]")
    with np.errstate(divide="ignore"):
        return bvn_cdf(ndtri(a), ndtri(b), r)


def default_correlation(pd_a: ArrayLike, pd_b: ArrayLike, rho: ArrayLike) -> Array:
    """Correlation of the two default events (zero-one indicators) implied by the asset correlation."""
    a = as_array("pd_a", np.atleast_1d(pd_a), 1)
    b = as_array("pd_b", np.atleast_1d(pd_b), 1)
    joint = joint_default_probability(a, b, rho)
    denom = np.sqrt(a * (1.0 - a) * b * (1.0 - b))
    if np.any(denom == 0.0):
        raise InputError("default correlation is undefined when a default probability is 0 or 1")
    out: Array = (joint - a * b) / denom
    return out


def validate_transition_matrix(matrix: ArrayLike, tol: float = 1e-3) -> Array:
    """A one-period transition matrix: one row per non-default rating, one column per state incl. default."""
    m = as_array("transition matrix", matrix, 2)
    if m.shape[1] != m.shape[0] + 1:
        raise InputError("a transition matrix needs one more column (default) than rows")
    return np.vstack([probability_row(f"transition matrix row {i + 1}", row, tol) for i, row in enumerate(m)])
