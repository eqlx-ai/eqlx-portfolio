"""MCP server: lets an AI agent measure credit portfolio risk on files in one folder.

Read-only by construction. The tools read loan files (CSV) and settings files (JSON) inside the
``--root`` folder given at start-up; paths are resolved, so ``..`` and symlinks cannot leave it.
Nothing is written and nothing is sent anywhere; results go back to the agent that asked. The
calculator tools take plain numbers and touch no file at all.

Start it with ``eqlx-portfolio mcp --root /path/to/folder`` (needs ``pip install 'eqlx-portfolio[mcp]'``).
"""

from __future__ import annotations

import functools
import json
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import config, data, engine, ratings, valuation, vasicek
from ._version import __version__
from .errors import InputError

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
CACHE_SIZE = 4
MAX_SETTINGS_BYTES = 5_000_000
MAX_NEW_LOANS = 1_000


F = TypeVar("F", bound=Callable[..., Any])


class OutsideRootError(ValueError):
    pass


def _explained(fn: F) -> F:
    """Pass the reason for a refused input on to the agent instead of a bare "error executing tool"."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except OutsideRootError as exc:
            raise ToolError(str(exc)) from exc
        except FileNotFoundError as exc:
            raise ToolError(f"file not found: {exc}") from exc
        except InputError as exc:
            raise ToolError(str(exc)) from exc
        except (IsADirectoryError, PermissionError) as exc:
            raise ToolError(f"cannot read: {type(exc).__name__}") from exc

    return wrapper  # type: ignore[return-value]


def _resolve(root: Path, path: str) -> Path:
    if not isinstance(path, str) or "\x00" in path:
        raise OutsideRootError("the path is not valid")
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise OutsideRootError(f"{path!r} is outside the allowed folder")
    if not p.exists():
        raise FileNotFoundError(path)
    return p


class _Runs:
    """Keeps the last few loan files and runs in memory so follow-up questions are fast."""

    def __init__(self, root: Path, max_rows: int) -> None:
        self.root = root
        self.max_rows = max_rows
        self.loans: OrderedDict[tuple[str, int, int], data.Loans] = OrderedDict()
        self.runs: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()

    def file(self, path: str, suffix: str) -> Path:
        p = _resolve(self.root, path)
        if not p.is_file():
            raise InputError(f"{path!r} is not a file")
        if p.suffix.lower() != suffix:
            raise InputError(f"{path!r} is not a {suffix} file")
        return p

    def get_loans(self, path: str) -> tuple[tuple[str, int, int], data.Loans]:
        p = self.file(path, ".csv")
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        if key not in self.loans:
            self.loans[key] = data.read_csv(p, max_rows=self.max_rows)
            while len(self.loans) > CACHE_SIZE:
                self.loans.popitem(last=False)
        self.loans.move_to_end(key)
        return key, self.loans[key]

    def settings(self, path: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
        p = self.file(path, ".json")
        if p.stat().st_size > MAX_SETTINGS_BYTES:
            raise InputError("the settings file is too large")
        raw = config.read_json(p)
        return config.with_overrides(raw, overrides) if overrides else raw

    def run(self, loans: str, settings: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
        key, lo = self.get_loans(loans)
        raw = self.settings(settings, overrides)
        run_key = (*key, json.dumps(raw, sort_keys=True))
        if run_key not in self.runs:
            self.runs[run_key] = engine.run(lo, config.from_dict({**raw, "top": data.MAX_ROWS}))
            while len(self.runs) > CACHE_SIZE:
                self.runs.popitem(last=False)
        self.runs.move_to_end(run_key)
        return self.runs[run_key]


def build_server(root: Path, max_rows: int = data.MAX_ROWS) -> MCPServer:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    runs = _Runs(root, max_rows)
    server = MCPServer(
        "eqlx-portfolio",
        version=__version__,
        instructions=(
            "Measures the credit risk of a loan portfolio as a whole: how much it could lose in a bad year, how "
            "much capital that takes, and which borrowers and sectors drive it. Files live in one local folder; "
            "paths are relative to it. A run needs a loan file (CSV: loan_id, exposure, pd, lgd, optional "
            "borrower_id, sector, lgd_sd) and a settings file (JSON) whose correlation block is the main "
            "assumption. Start with `list_files` and `describe_loans`, then `portfolio_risk`; use "
            "`risk_contributions` to see who drives the risk and `what_if_new_loans` to test a new loan. "
            "`joint_default`, `joint_migration`, `exposure_value` and `large_portfolio_benchmark` are calculators "
            "that need no file. Everything is deterministic: a simulation uses a fixed seed, so the same input "
            "gives the same numbers. The tool does not estimate pd, lgd or correlations. Quote the numbers the "
            "tools return, say which settings produced them, and for simulated value-at-risk mention the band."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    @_explained
    def list_files(path: str = ".") -> list[dict[str, Any]]:
        """List the loan files (.csv) and settings files (.json) under a path in the allowed folder."""
        p = _resolve(root, path)
        files = [p] if p.is_file() else sorted(p.rglob("*"))
        out = []
        for f in files:
            if f.suffix.lower() not in (".csv", ".json") or not f.is_file():
                continue
            real = f.resolve()
            if root in real.parents:
                out.append({"path": str(real.relative_to(root)), "bytes": real.stat().st_size})
        return out

    @server.tool(annotations=READ_ONLY)
    @_explained
    def describe_loans(loans: str) -> dict[str, Any]:
        """What a loan file contains before any correlation assumption: loans, borrowers, sectors, exposure,
        expected loss, the largest borrowers and how concentrated the exposure is."""
        return engine.describe(runs.get_loans(loans)[1])

    @server.tool(annotations=READ_ONLY)
    @_explained
    def portfolio_risk(loans: str, settings: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Loss distribution of the whole portfolio: expected loss, standard deviation, and at each confidence
        level value-at-risk, expected shortfall and economic capital; plus concentration by sector, the ten
        largest risk contributors, the checks that were run and the steps taken.

        `settings` is a JSON file: method ("simulation" or "creditriskplus"), confidence levels, correlation
        (model "single" with rho, "basel_corporate", or "sector" with within/between), scenarios and seed.
        `overrides` replaces single settings by dotted key, e.g. {"correlation.rho": 0.25, "scenarios": 400000}.
        """
        res = dict(runs.run(loans, settings, overrides))
        res["top_contributors"] = res["top_contributors"][:10]
        return res

    @server.tool(annotations=READ_ONLY)
    @_explained
    def risk_contributions(
        loans: str,
        settings: str,
        overrides: dict[str, Any] | None = None,
        sector: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Borrowers ranked by their share of portfolio risk (their average loss in the worst outcomes), with
        exposure share next to risk share. A borrower whose risk share is far above its exposure share is
        where the concentration is. `sector` filters; `limit` is 1 to 1000."""
        if isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise InputError("limit must be between 1 and 1000")
        res = runs.run(loans, settings, overrides)
        rows = res["top_contributors"]
        if sector is not None:
            known = sorted({r["sector"] for r in rows})
            if sector not in known:
                raise InputError(f"no sector {sector!r} in the loan file; sectors: {', '.join(known)}")
            rows = [r for r in rows if r["sector"] == sector]
        return {
            "confidence": res["contributions_at_confidence"],
            "expected_shortfall": res["levels"][-1]["expected_shortfall"],
            "borrowers_matching": len(rows),
            "borrowers": rows[:limit],
        }

    @server.tool(annotations=READ_ONLY)
    @_explained
    def what_if_new_loans(
        loans: str, settings: str, new_loans: list[dict[str, Any]], overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """What adding loans to new borrowers does: expected loss, value-at-risk, expected shortfall and
        economic capital before and after, the capital taken per unit of new exposure, and the change in
        concentration. Each new loan is {"loan_id", "exposure", "pd", "lgd"} with optional "borrower_id",
        "sector", "lgd_sd". Nothing is saved; the loan file is not changed."""
        if not new_loans or len(new_loans) > MAX_NEW_LOANS:
            raise InputError(f"new_loans needs between 1 and {MAX_NEW_LOANS} loans")
        _, lo = runs.get_loans(loans)
        raw = runs.settings(settings, overrides)
        return engine.what_if(lo, data.from_records(new_loans), config.from_dict(raw))

    @server.tool(annotations=READ_ONLY)
    @_explained
    def joint_default(pd_a: float, pd_b: float, asset_correlation: float) -> dict[str, float | None]:
        """Probability that two borrowers both default in the period, from their default probabilities and
        the correlation of their asset values; also what it would be if they were independent, and the
        correlation of the two default events (null when a default probability is 0 or 1)."""
        res = engine.pair(pd_a, pd_b, asset_correlation)
        return {k: res[k] for k in ("joint_default_probability", "if_independent", "default_correlation")}

    @server.tool(annotations=READ_ONLY)
    @_explained
    def joint_migration(probs_a: list[float], probs_b: list[float], asset_correlation: float) -> dict[str, Any]:
        """Joint probabilities of the year-end ratings of two borrowers. Each list gives one borrower's
        probabilities of ending in each state, best rating first and default last. Returns each borrower's
        asset return thresholds (null where a state has no probability and the threshold is infinite) and the
        table of joint probabilities (rows: borrower A)."""

        def finite(values: Any) -> list[float | None]:
            return [float(v) if np.isfinite(v) else None for v in values]

        return {
            "thresholds_a": finite(ratings.migration_thresholds(probs_a)),
            "thresholds_b": finite(ratings.migration_thresholds(probs_b)),
            "joint_probabilities": ratings.joint_migration(probs_a, probs_b, asset_correlation).tolist(),
        }

    @server.tool(annotations=READ_ONLY)
    @_explained
    def exposure_value(
        probs: list[float], values: list[float], value_sds: list[float] | None = None
    ) -> dict[str, float]:
        """Mean and standard deviation of one exposure's value at the horizon. `probs` are the probabilities
        of each year-end state (best first, default last), `values` its value in each state, `value_sds` the
        optional uncertainty of the value inside each state (for example of recovery in default)."""
        s = valuation.value_stats(probs, values, value_sds)
        return {"mean": s.mean, "standard_deviation": s.sd, "variance": s.variance}

    @server.tool(annotations=READ_ONLY)
    @_explained
    def large_portfolio_benchmark(
        pd: float, lgd: float, asset_correlation: float, confidence: float = 0.999
    ) -> dict[str, float]:
        """Loss rate of a perfectly fine-grained portfolio of identical loans in a bad year (the one-factor
        formula behind the Basel risk weights, without the maturity adjustment): default rate at the
        confidence level, loss rate, expected loss rate and the difference (unexpected loss)."""
        if not 0.0 <= lgd <= 1.0:
            raise InputError("lgd must lie in [0, 1]")
        rate = float(vasicek.conditional_pd(np.array([pd]), asset_correlation, confidence)[0])
        return {
            "default_rate_at_confidence": rate,
            "loss_rate_at_confidence": rate * lgd,
            "expected_loss_rate": pd * lgd,
            "unexpected_loss_rate": (rate - pd) * lgd,
        }

    return server
