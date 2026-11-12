"""Input validation shared by the calculation modules. Nothing is clipped or repaired silently."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import InputError

Array = NDArray[np.float64]
MAX_MAGNITUDE = 1e15  # amounts beyond this lose whole units in double precision and overflow in squares


def as_array(name: str, values: ArrayLike, ndim: int) -> Array:
    try:
        a = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f"{name} must be numeric") from exc
    if ndim == 1 and a.ndim == 0:
        a = a.reshape(1)
    if a.ndim != ndim:
        raise InputError(f"{name} must have {ndim} dimension(s)")
    if a.size == 0:
        raise InputError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise InputError(f"{name} contains NaN or infinite values")
    if np.any(np.abs(a) > MAX_MAGNITUDE):
        raise InputError(f"{name} contains values beyond {MAX_MAGNITUDE:g}")
    return a


def as_1d(name: str, values: ArrayLike) -> Array:
    return as_array(name, values, 1)


def broadcast(name: str, values: ArrayLike, n: int) -> Array:
    """A scalar is repeated ``n`` times; an array must have exactly ``n`` entries."""
    a = as_1d(name, values)
    if a.size == 1 and n != 1:
        return np.full(n, float(a[0]))
    if a.size != n:
        raise InputError(f"{name} has {a.size} entries, expected {n}")
    return a


def in_unit_interval(name: str, a: Array) -> Array:
    if np.any(a < 0.0) or np.any(a > 1.0):
        raise InputError(f"{name} must lie in [0, 1]")
    return a


def non_negative(name: str, a: Array) -> Array:
    if np.any(a < 0.0):
        raise InputError(f"{name} must not be negative")
    return a


def probability_row(name: str, values: ArrayLike, tol: float = 1e-3) -> Array:
    """Probabilities of mutually exclusive outcomes.

    Must sum to one within ``tol`` (published tables are rounded) and are then normalised exactly.
    """
    p = in_unit_interval(name, as_1d(name, values))
    total = float(p.sum())
    if abs(total - 1.0) > tol:
        raise InputError(f"{name} sums to {total:.6f}, expected 1")
    return p / total


def correlation(name: str, rho: float) -> float:
    try:
        r = float(rho)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f"{name} must be a number") from exc
    if not np.isfinite(r) or r < -1.0 or r > 1.0:
        raise InputError(f"{name} must lie in [-1, 1]")
    return r


def confidence_levels(levels: ArrayLike) -> Array:
    q = as_1d("confidence", levels)
    if np.any(q <= 0.0) or np.any(q >= 1.0):
        raise InputError("confidence levels must lie strictly between 0 and 1")
    return np.unique(q)
