"""Loan files: one row per loan, grouped into borrowers before any risk is computed."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._checks import Array
from .errors import InputError

REQUIRED = ("loan_id", "exposure", "pd", "lgd")
OPTIONAL = ("borrower_id", "sector", "lgd_sd")
DEFAULT_SECTOR = "all"
MAX_ROWS = 2_000_000


@dataclass(frozen=True)
class Loans:
    loan_id: tuple[str, ...]
    borrower_id: tuple[str, ...]
    sector: tuple[str, ...]
    exposure: Array
    pd: Array
    lgd: Array
    lgd_sd: Array | None

    def __len__(self) -> int:
        return len(self.loan_id)


@dataclass(frozen=True)
class Borrowers:
    """Loans of one borrower default together, so risk is computed per borrower."""

    borrower_id: tuple[str, ...]
    sector_names: tuple[str, ...]
    sector: NDArray[np.intp]  # index into sector_names, per borrower
    exposure: Array
    amount: Array  # expected loss if the borrower defaults: sum of exposure x lgd
    pd: Array
    lgd: Array  # amount / exposure
    lgd_sd: Array | None
    loans: NDArray[np.intp]  # number of loans per borrower

    def __len__(self) -> int:
        return len(self.borrower_id)


def _number(row: int, name: str, raw: Any) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        raise InputError(f"row {row}: {name} is not a number ({raw!r})") from None
    if not np.isfinite(value):
        raise InputError(f"row {row}: {name} is not a finite number")
    return value


def from_records(records: Iterable[Mapping[str, Any]], max_rows: int = MAX_ROWS) -> Loans:
    """Build a loan table from dictionaries with the same keys as the CSV columns."""
    ids: list[str] = []
    borrowers: list[str] = []
    sectors: list[str] = []
    exposure: list[float] = []
    pd: list[float] = []
    lgd: list[float] = []
    lgd_sd: list[float] = []
    any_sd = False
    seen: set[str] = set()
    for row, rec in enumerate(records, start=2):  # row 1 is the header of a CSV file
        if len(ids) >= max_rows:
            raise InputError(f"the file has more than {max_rows} loans")
        missing = [c for c in REQUIRED if rec.get(c) is None or str(rec.get(c)).strip() == ""]
        if missing:
            raise InputError(f"row {row}: missing {', '.join(missing)}")
        loan_id = str(rec["loan_id"]).strip()
        if loan_id in seen:
            raise InputError(f"row {row}: loan_id {loan_id!r} appears more than once")
        seen.add(loan_id)
        e = _number(row, "exposure", rec["exposure"])
        p = _number(row, "pd", rec["pd"])
        g = _number(row, "lgd", rec["lgd"])
        if e <= 0.0:
            raise InputError(f"row {row}: exposure must be positive")
        if not 0.0 <= p < 1.0:
            raise InputError(f"row {row}: pd must be at least 0 and below 1 (leave defaulted loans out)")
        if not 0.0 <= g <= 1.0:
            raise InputError(f"row {row}: lgd must lie in [0, 1]")
        raw_sd = rec.get("lgd_sd")
        sd = 0.0
        if raw_sd is not None and str(raw_sd).strip() != "":
            sd = _number(row, "lgd_sd", raw_sd)
            if sd < 0.0:
                raise InputError(f"row {row}: lgd_sd must not be negative")
            if sd > 0.0 and sd * sd >= g * (1.0 - g):
                raise InputError(f"row {row}: lgd_sd is too large for a loss rate between 0 and 1 with mean {g}")
            any_sd = any_sd or sd > 0.0
        b = rec.get("borrower_id")
        s = rec.get("sector")
        ids.append(loan_id)
        borrowers.append(loan_id if b is None or str(b).strip() == "" else str(b).strip())
        sectors.append(DEFAULT_SECTOR if s is None or str(s).strip() == "" else str(s).strip())
        exposure.append(e)
        pd.append(p)
        lgd.append(g)
        lgd_sd.append(sd)
    if not ids:
        raise InputError("the file has no loans")
    return Loans(
        tuple(ids),
        tuple(borrowers),
        tuple(sectors),
        np.array(exposure),
        np.array(pd),
        np.array(lgd),
        np.array(lgd_sd) if any_sd else None,
    )


def read_csv(path: str | Path, max_rows: int = MAX_ROWS) -> Loans:
    """Read a loan file. Columns: loan_id, exposure, pd, lgd; optional borrower_id, sector, lgd_sd."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InputError("the file is empty")
        names = [n.strip() for n in reader.fieldnames]
        missing = [c for c in REQUIRED if c not in names]
        if missing:
            raise InputError(f"missing column(s): {', '.join(missing)}")
        reader.fieldnames = names
        return from_records(reader, max_rows)


def group(loans: Loans) -> Borrowers:
    """One line per borrower. All loans of a borrower must share its pd and sector."""
    order: dict[str, int] = {}
    for b in loans.borrower_id:
        order.setdefault(b, len(order))
    idx = np.array([order[b] for b in loans.borrower_id], dtype=np.intp)
    n = len(order)
    first = np.full(n, -1, dtype=np.intp)
    for row, i in enumerate(idx):
        if first[i] < 0:
            first[i] = row
    pd = loans.pd[first]
    if np.any(np.abs(loans.pd - pd[idx]) > 1e-12):
        bad = loans.borrower_id[int(np.argmax(np.abs(loans.pd - pd[idx]) > 1e-12))]
        raise InputError(f"borrower {bad!r} has loans with different pd; a borrower defaults as a whole")
    sector_of = [loans.sector[i] for i in first]
    for row, i in enumerate(idx):
        if loans.sector[row] != sector_of[i]:
            raise InputError(f"borrower {loans.borrower_id[row]!r} has loans in different sectors")
    names = tuple(sorted(set(sector_of)))
    lookup = {s: k for k, s in enumerate(names)}
    exposure = np.bincount(idx, weights=loans.exposure, minlength=n)
    amount = np.bincount(idx, weights=loans.exposure * loans.lgd, minlength=n)
    lgd_sd = None
    if loans.lgd_sd is not None:
        # One recovery outcome per borrower: its loans share the same draw, so the standard
        # deviations add up in proportion to exposure.
        lgd_sd = np.bincount(idx, weights=loans.exposure * loans.lgd_sd, minlength=n) / exposure
    return Borrowers(
        tuple(order),
        names,
        np.array([lookup[s] for s in sector_of], dtype=np.intp),
        exposure,
        amount,
        pd,
        amount / exposure,
        lgd_sd,
        np.bincount(idx, minlength=n).astype(np.intp),
    )
