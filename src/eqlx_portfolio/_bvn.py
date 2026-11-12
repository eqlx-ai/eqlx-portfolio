"""Bivariate standard normal distribution function.

Uses the identity (Plackett 1954; Drezner and Wesolowsky 1990)

    Phi2(h, k; rho) = Phi(h) Phi(k) + 1/(2 pi) * integral_0^asin(rho) exp(-(h^2 + k^2 - 2 h k sin t) / (2 cos^2 t)) dt

with Gauss-Legendre quadrature. The integrand is smooth except as |rho| -> 1, where the two
limits are known in closed form; close to those limits the range is split so the nodes
concentrate near the end point.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from ._checks import Array

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(48)
_CLIP = 38.0  # Phi(-38) is exactly 0.0 in double precision
_TWO_PI = 2.0 * np.pi


def _integral(h: Array, k: Array, lo: Array, hi: Array) -> Array:
    """Integral of the Plackett integrand over [lo, hi] for each element."""
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    t = mid[..., None] + half[..., None] * _NODES
    s = np.sin(t)
    c2 = np.maximum(1.0 - s * s, 1e-300)
    hh = h[..., None]
    kk = k[..., None]
    f = np.exp(-(hh * hh + kk * kk - 2.0 * hh * kk * s) / (2.0 * c2))
    out: Array = half * (f @ _WEIGHTS)
    return out


def bvn_cdf(h: ArrayLike, k: ArrayLike, rho: ArrayLike) -> Array:
    """P(X <= h, Y <= k) for standard normal X, Y with correlation ``rho``. Arguments broadcast."""
    hb, kb, rb = np.broadcast_arrays(
        np.asarray(h, dtype=np.float64), np.asarray(k, dtype=np.float64), np.asarray(rho, dtype=np.float64)
    )
    hc = np.clip(hb, -_CLIP, _CLIP)
    kc = np.clip(kb, -_CLIP, _CLIP)
    r = np.clip(rb, -1.0, 1.0)
    ph = ndtr(hc)
    pk = ndtr(kc)
    a = np.arcsin(r)
    # Split [0, asin(rho)] into four panels that shrink towards the end point: the integrand
    # sharpens there when |rho| is close to one.
    total = np.zeros(hc.shape)
    for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)):
        total = total + _integral(hc, kc, a * lo, a * hi)
    out = ph * pk + total / _TWO_PI
    out = np.where(r >= 1.0, np.minimum(ph, pk), out)
    out = np.where(r <= -1.0, np.maximum(ph + pk - 1.0, 0.0), out)
    lower = np.maximum(ph + pk - 1.0, 0.0)
    upper = np.minimum(ph, pk)
    result: Array = np.clip(out, lower, upper)
    return result
