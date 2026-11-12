"""Seeded Monte Carlo simulation of portfolio outcomes.

Every scenario draws the common factors and each borrower's own shock, finds the borrower's
year-end state from its thresholds, and adds up the outcome. The random numbers come from
NumPy's PCG64 generator with the seed given, in fixed-size batches, so the same input, seed
and batch size give the same result on every run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaincinv, ndtri
from scipy.stats import binom

from ._bvn import bvn_cdf
from ._checks import Array, as_array, broadcast, confidence_levels, in_unit_interval, non_negative
from .errors import InputError
from .factors import FactorModel

BATCH_CELLS = 2_000_000  # scenarios x borrowers generated at a time; part of the reproducibility contract
MAX_SCENARIOS = 20_000_000
EXACT_SD_MAX_BORROWERS = 1_500


@dataclass(frozen=True)
class TailShare:
    confidence: float
    value_at_risk: float
    expected_shortfall: float
    contributions: Array  # per borrower; sums to expected_shortfall


@dataclass(frozen=True)
class Simulation:
    totals: Array  # one portfolio outcome per scenario, in scenario order
    totals_excluding: Array | None  # the same scenarios without the flagged borrowers
    tails: tuple[TailShare, ...]
    scenarios: int
    seed: int


def _check_run(scenarios: int, seed: int) -> None:
    if isinstance(scenarios, bool) or not isinstance(scenarios, int) or scenarios < 100:
        raise InputError("scenarios must be a whole number, at least 100")
    if scenarios > MAX_SCENARIOS:
        raise InputError(f"scenarios must not exceed {MAX_SCENARIOS}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise InputError("seed must be a non-negative whole number")


def order_index(scenarios: int, confidence: float) -> int:
    """Zero-based position, in ascending order, of the scenario taken as the ``confidence`` quantile."""
    return min(scenarios - 1, max(0, int(np.ceil(confidence * scenarios - 1e-9)) - 1))


def _beta_parameters(mean: Array, sd: Array) -> tuple[Array, Array]:
    var = sd * sd
    limit = mean * (1.0 - mean)
    if np.any((sd > 0.0) & (var >= limit)):
        raise InputError("lgd_sd is too large for a loss rate between 0 and 1 with that mean")
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = np.where(sd > 0.0, limit / np.where(var > 0.0, var, 1.0) - 1.0, 1.0)
    return mean * nu, (1.0 - mean) * nu


def simulate(
    model: FactorModel,
    thresholds: ArrayLike,
    outcomes: ArrayLike,
    *,
    scenarios: int,
    seed: int,
    tail_levels: ArrayLike | None = None,
    exclude: NDArray[np.bool_] | None = None,
    default_loss_rate_sd: ArrayLike | None = None,
    default_loss_rate_mean: ArrayLike | None = None,
) -> Simulation:
    """Simulate the portfolio total of state-dependent outcomes.

    ``thresholds[i]`` are borrower ``i``'s asset return thresholds as returned by
    ``ratings.migration_thresholds`` (descending; the last is the default threshold) and
    ``outcomes[i, s]`` is what the borrower contributes to the total in state ``s`` (best state
    first, default last): a loss amount in default mode, a value in migration mode.

    ``tail_levels`` asks for value-at-risk, expected shortfall and each borrower's share of the
    expected shortfall at those confidence levels, treating large totals as bad. The shares are
    the borrower's average outcome over the worst scenarios and add up to the expected shortfall.

    ``default_loss_rate_sd`` makes the outcome in default uncertain: it is scaled by a
    beta-distributed loss rate with mean ``default_loss_rate_mean`` and this standard deviation.
    """
    _check_run(scenarios, seed)
    thr = as_array_allow_inf("thresholds", thresholds)
    out = as_array("outcomes", outcomes, 2)
    n, states = out.shape
    if model.n != n or thr.shape != (n, states - 1):
        raise InputError("thresholds, outcomes and the factor model must describe the same borrowers and states")
    if exclude is not None and (exclude.shape != (n,) or exclude.dtype != np.bool_):
        raise InputError("exclude must be one true/false flag per borrower")
    levels = None if tail_levels is None else confidence_levels(tail_levels)

    random_lgd = default_loss_rate_sd is not None
    if random_lgd:
        if default_loss_rate_mean is None:
            raise InputError("default_loss_rate_mean is required with default_loss_rate_sd")
        assert default_loss_rate_sd is not None
        lgd_mean = in_unit_interval(
            "default_loss_rate_mean", broadcast("default_loss_rate_mean", default_loss_rate_mean, n)
        )
        lgd_sd = non_negative("default_loss_rate_sd", broadcast("default_loss_rate_sd", default_loss_rate_sd, n))
        alpha, beta = _beta_parameters(lgd_mean, lgd_sd)
        uncertain = lgd_sd > 0.0
        if np.any(uncertain & (lgd_mean <= 0.0)):
            raise InputError("lgd_sd needs a positive mean loss rate")

    # Flagged borrowers draw their own shocks from a second stream, and the batch size ignores
    # them: the totals without them are then identical to a run of the portfolio without them.
    main = np.ones(n, dtype=np.bool_) if exclude is None else ~exclude
    n_main = int(main.sum())
    if n_main == 0:
        raise InputError("exclude must leave at least one borrower")
    batch = max(1, BATCH_CELLS // n_main)
    k = model.loadings.shape[1]
    load_t = np.ascontiguousarray(model.loadings.T)
    cols = np.arange(n)

    def draw(rng: np.random.Generator, side: np.random.Generator, m: int, normal: bool) -> Array:
        if exclude is None:
            return rng.standard_normal((m, n)) if normal else rng.random((m, n))
        x = np.empty((m, n))
        x[:, main] = rng.standard_normal((m, n_main)) if normal else rng.random((m, n_main))
        x[:, exclude] = side.standard_normal((m, n - n_main)) if normal else side.random((m, n - n_main))
        return x

    def run(cutoffs: list[tuple[float, float]] | None) -> tuple[Array, Array | None, list[Array]]:
        rng = np.random.Generator(np.random.PCG64(seed))
        side = np.random.Generator(np.random.PCG64([seed, 1]))
        totals = np.empty(scenarios)
        without = np.empty(scenarios) if exclude is not None else None
        sums = [np.zeros(n) for _ in (cutoffs or [])]
        done = 0
        while done < scenarios:
            m = min(batch, scenarios - done)
            f = rng.standard_normal((m, k))
            e = draw(rng, side, m, True)
            r = f @ load_t + e * model.idio
            state = np.zeros((m, n), dtype=np.intp)
            for j in range(states - 1):
                state += r < thr[:, j]
            x = out[cols, state]
            if random_lgd:
                u = draw(rng, side, m, False)
                hit = (state == states - 1) & uncertain
                if hit.any():
                    rows, cs = np.nonzero(hit)
                    rate = betaincinv(alpha[cs], beta[cs], u[rows, cs])
                    x[rows, cs] = out[cs, states - 1] * rate / lgd_mean[cs]
            x = np.ascontiguousarray(x)  # fixed memory order, so sums round the same way in every run
            tot = x.sum(axis=1)
            totals[done : done + m] = tot
            if without is not None:
                without[done : done + m] = np.ascontiguousarray(x[:, main]).sum(axis=1)
            for (cut, weight_at_cut), acc in zip(cutoffs or [], sums, strict=True):
                above = tot > cut
                at = tot == cut
                if above.any():
                    acc += x[above].sum(axis=0)
                if weight_at_cut > 0.0 and at.any():
                    acc += weight_at_cut * x[at].sum(axis=0)
            done += m
        return totals, without, sums

    totals, without, _ = run(None)
    tails: list[TailShare] = []
    if levels is not None:
        ordered = np.sort(totals)
        cutoffs: list[tuple[float, float]] = []
        meta: list[tuple[float, float, float, int]] = []
        for q in levels:
            idx = order_index(scenarios, float(q))
            var = float(ordered[idx])
            tail_n = scenarios - idx
            n_above = int(scenarios - np.searchsorted(ordered, var, side="right"))
            n_at = int(np.searchsorted(ordered, var, side="right") - np.searchsorted(ordered, var, side="left"))
            weight = (tail_n - n_above) / n_at
            es = float(ordered[idx:].mean())
            cutoffs.append((var, weight))
            meta.append((float(q), var, es, tail_n))
        _, _, sums = run(cutoffs)
        for (q, var, es, tail_n), acc in zip(meta, sums, strict=True):
            tails.append(TailShare(q, var, es, acc / tail_n))
    return Simulation(totals, without, tuple(tails), scenarios, seed)


def as_array_allow_inf(name: str, values: ArrayLike) -> Array:
    """Thresholds may be infinite (a state with zero probability at the edge)."""
    try:
        a = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be numeric") from exc
    if a.ndim != 2 or a.size == 0 or np.any(np.isnan(a)):
        raise InputError(f"{name} must be a two-dimensional table without missing values")
    return a


def default_thresholds(pd: ArrayLike) -> Array:
    """Default-mode thresholds: one column, the asset return below which the borrower defaults."""
    p = in_unit_interval("pd", as_array("pd", pd, 1))
    with np.errstate(divide="ignore"):
        out: Array = ndtri(p)[:, None]
    return out


@dataclass(frozen=True)
class Quantile:
    confidence: float
    value: float
    ci_low: float
    ci_high: float
    expected_shortfall: float


def quantiles(totals: Array, levels: ArrayLike, ci: float = 0.95) -> list[Quantile]:
    """Quantiles of the simulated totals with a distribution-free confidence band.

    The band comes from order statistics: the number of scenarios below the true quantile is
    binomial, so two sorted scenarios bracket it with probability ``ci`` (Technical Document,
    appendix B).
    """
    x = np.sort(totals)
    n = x.size
    out = []
    for q in confidence_levels(levels):
        idx = order_index(n, float(q))
        lo = int(binom.ppf((1.0 - ci) / 2.0, n, q))
        hi = int(binom.ppf(1.0 - (1.0 - ci) / 2.0, n, q))
        out.append(
            Quantile(
                float(q),
                float(x[idx]),
                float(x[min(max(lo - 1, 0), n - 1)]),
                float(x[min(max(hi, 0), n - 1)]),
                float(x[idx:].mean()),
            )
        )
    return out


def exact_loss_moments(
    amount: ArrayLike, pd: ArrayLike, model: FactorModel, amount_sd: ArrayLike | None = None
) -> tuple[float, float]:
    """Exact mean and standard deviation of default-mode loss, pair by pair, without simulation.

    ``amount`` is the loss if the borrower defaults; ``amount_sd`` its standard deviation given
    default. Needs a joint default probability for every pair, so it is limited to
    ``EXACT_SD_MAX_BORROWERS`` borrowers.
    """
    p = in_unit_interval("pd", as_array("pd", pd, 1))
    n = p.size
    a = non_negative("amount", broadcast("amount", amount, n))
    if model.n != n:
        raise InputError("the factor model must describe the same borrowers")
    if n > EXACT_SD_MAX_BORROWERS:
        raise InputError(f"the exact standard deviation is limited to {EXACT_SD_MAX_BORROWERS} borrowers")
    mean = float(a @ p)
    var = float((a * a) @ (p * (1.0 - p)))
    if amount_sd is not None:
        s = non_negative("amount_sd", broadcast("amount_sd", amount_sd, n))
        var += float((s * s) @ p)
    if n > 1:
        with np.errstate(divide="ignore"):
            z = ndtri(p)
        i, j = np.triu_indices(n, 1)
        rho = np.clip((model.loadings[i] * model.loadings[j]).sum(axis=1), -1.0, 1.0)
        joint = bvn_cdf(z[i], z[j], rho)
        var += 2.0 * float((a[i] * a[j]) @ (joint - p[i] * p[j]))
    return mean, float(np.sqrt(max(var, 0.0)))
