"""How borrowers move together: asset returns driven by common factors.

Each borrower's standard normal asset return is ``loadings @ F + idio * e`` with independent
standard normal factors ``F`` and a borrower-specific ``e``. The asset correlation of two
different borrowers is the dot product of their loadings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._checks import Array, as_array, broadcast, in_unit_interval
from .errors import InputError


@dataclass(frozen=True)
class FactorModel:
    loadings: Array  # (borrowers, factors)
    idio: Array  # (borrowers,)

    @property
    def n(self) -> int:
        return int(self.loadings.shape[0])

    def correlation(self, i: int, j: int) -> float:
        return 1.0 if i == j else float(self.loadings[i] @ self.loadings[j])

    def correlation_matrix(self) -> Array:
        m: Array = self.loadings @ self.loadings.T
        np.fill_diagonal(m, 1.0)
        return m


def _model(loadings: Array) -> FactorModel:
    r2 = (loadings * loadings).sum(axis=1)
    if np.any(r2 > 1.0 + 1e-9):
        raise InputError("a borrower's systematic share exceeds 100%")
    return FactorModel(loadings, np.sqrt(np.clip(1.0 - r2, 0.0, 1.0)))


def _root(matrix: Array, name: str) -> Array:
    """A matrix ``r`` with ``r @ r.T == matrix``, chosen the same way on every machine.

    The Cholesky factor is unique, so the scenarios do not depend on the linear algebra library.
    A matrix that is only positive semi-definite (perfectly correlated factors) has no Cholesky
    factor; its eigenvectors are used instead, each with its largest entry made positive.
    """
    values, vectors = np.linalg.eigh(matrix)
    if values.min() < -1e-8:
        raise InputError(f"{name} is not positive semi-definite")
    if values.min() > 1e-10:
        out: Array = np.linalg.cholesky(matrix)
        return out
    lead = np.abs(vectors).argmax(axis=0)
    signs = np.sign(vectors[lead, np.arange(vectors.shape[1])])
    signs[signs == 0.0] = 1.0
    out = vectors * signs * np.sqrt(np.clip(values, 0.0, None))
    return out


def single_factor(rho: ArrayLike, n: int) -> FactorModel:
    """One common factor. ``rho`` is the asset correlation, one number or one per borrower.

    With borrower-specific values the correlation of borrowers i and j is ``sqrt(rho_i * rho_j)``.
    """
    r = in_unit_interval("rho", broadcast("rho", rho, n))
    return _model(np.sqrt(r)[:, None])


def sector_factors(sector: NDArray[np.intp], within: ArrayLike, between: ArrayLike) -> FactorModel:
    """One factor per sector.

    ``sector[i]`` is the sector number of borrower ``i``; ``within[s]`` is the asset correlation
    of two borrowers inside sector ``s``; ``between`` is the correlation matrix of the sector
    factors. Borrowers in sectors ``s`` and ``t`` then have asset correlation
    ``sqrt(within[s] * within[t]) * between[s, t]``.
    """
    w = in_unit_interval("within", as_array("within", within, 1))
    b = as_array("between", between, 2)
    k = w.size
    if b.shape != (k, k) or not np.allclose(b, b.T) or not np.allclose(np.diag(b), 1.0):
        raise InputError("between must be a symmetric sector-by-sector matrix with ones on the diagonal")
    if np.any(np.abs(b) > 1.0 + 1e-12):
        raise InputError("between-sector correlations must lie in [-1, 1]")
    idx = np.asarray(sector, dtype=np.intp)
    if idx.ndim != 1 or idx.size == 0 or idx.min() < 0 or idx.max() >= k:
        raise InputError("sector numbers must lie between 0 and the number of sectors minus one")
    root = _root(b, "the between-sector correlation matrix")
    return _model(np.sqrt(w)[idx, None] * root[idx])


def from_correlation_matrix(corr: ArrayLike) -> FactorModel:
    """A full asset correlation matrix, for small portfolios."""
    c = as_array("correlations", corr, 2)
    n = c.shape[0]
    if c.shape != (n, n) or not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
        raise InputError("correlations must be a symmetric matrix with ones on the diagonal")
    return FactorModel(_root(c, "the correlation matrix"), np.zeros(n))
