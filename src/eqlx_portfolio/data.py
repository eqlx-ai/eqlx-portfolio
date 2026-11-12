"""Loan files: one row per loan, grouped into borrowers before any risk is computed."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._checks import MAX_MAGNITUDE, Array
from .errors import InputError

REQUIRED = ("loan_id", "exposure", "pd", "lgd")
OPTIONAL = ("borrower_id", "sector", "lgd_sd")
COLUMNS = REQUIRED + OPTIONAL
DEFAULT_SECTOR = "all"
MAX_ROWS = 2_000_000
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


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
    """Plain decimal notation only: no thousands separators, underscores or non-ASCII digits."""
    if isinstance(raw, bool):
        raise InputError(f"row {row}: {name} is not a number ({raw!r})")
    if isinstance(raw, (int, float)):
        try:
            value = float(raw)
        except OverflowError:
            raise InputError(f"row {row}: {name} is too large") from None
    else:
        text = str(raw).strip()
        if not _NUMBER.fullmatch(text):
            raise InputError(f"row {row}: {name} is not a number ({raw!r})")
        value = float(text)
    if not np.isfinite(value):
        raise InputError(f"row {row}: {name} is not a finite number")
    if abs(value) > MAX_MAGNITUDE:
        raise InputError(f"row {row}: {name} is too large")
    return value


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _build(rows: Iterable[tuple[int, Mapping[str, Any]]], max_rows: int) -> Loans:
    ids: list[str] = []
    borrowers: list[str | None] = []
    sectors: list[str] = []
    exposure: list[float] = []
    pd: list[float] = []
    lgd: list[float] = []
    lgd_sd: list[float] = []
    any_sd = False
    seen: set[str] = set()
    for row, rec in rows:
        if len(ids) >= max_rows:
            raise InputError(f"the file has more than {max_rows} loans")
        if None in rec:
            raise InputError(f"row {row}: more fields than the header has columns (a thousands separator?)")
        unknown = sorted(str(k) for k in rec if k not in COLUMNS)
        if unknown:
            raise InputError(f"row {row}: unknown column(s): {', '.join(unknown)}; allowed: {', '.join(COLUMNS)}")
        missing = [c for c in REQUIRED if _blank(rec.get(c))]
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
        sd = 0.0
        if not _blank(rec.get("lgd_sd")):
            sd = _number(row, "lgd_sd", rec["lgd_sd"])
            if sd < 0.0:
                raise InputError(f"row {row}: lgd_sd must not be negative")
            if sd > 0.0 and sd * sd >= g * (1.0 - g):
                raise InputError(f"row {row}: lgd_sd is too large for a loss rate between 0 and 1 with mean {g}")
            any_sd = any_sd or sd > 0.0
        ids.append(loan_id)
        borrowers.append(None if _blank(rec.get("borrower_id")) else str(rec["borrower_id"]).strip())
        sectors.append(DEFAULT_SECTOR if _blank(rec.get("sector")) else str(rec["sector"]).strip())
        exposure.append(e)
        pd.append(p)
        lgd.append(g)
        lgd_sd.append(sd)
    if not ids:
        raise InputError("the file has no loans")
    # A loan without a borrower_id is its own borrower. Its loan_id must not double as another
    # loan's borrower_id, or the two would be merged by accident.
    named = {b for b in borrowers if b is not None}
    resolved: list[str] = []
    for loan_id, b in zip(ids, borrowers, strict=True):
        if b is None and loan_id in named:
            raise InputError(f"loan_id {loan_id!r} has no borrower_id but is used as the borrower_id of another loan")
        resolved.append(loan_id if b is None else b)
    return Loans(
        tuple(ids),
        tuple(resolved),
        tuple(sectors),
        np.array(exposure),
        np.array(pd),
        np.array(lgd),
        np.array(lgd_sd) if any_sd else None,
    )


def from_records(records: Iterable[Mapping[str, Any]], max_rows: int = MAX_ROWS) -> Loans:
    """Build a loan table from dictionaries with the same keys as the CSV columns."""
    return _build(enumerate(records, start=1), max_rows)


def _check_header(names: Sequence[str]) -> None:
    missing = [c for c in REQUIRED if c not in names]
    if missing:
        raise InputError(f"missing column(s): {', '.join(missing)}")
    duplicate = sorted({n for n in names if names.count(n) > 1})
    if duplicate:
        raise InputError(f"column(s) appear more than once: {', '.join(duplicate)}")
    unknown = sorted(n for n in names if n not in COLUMNS)
    if unknown:
        raise InputError(f"unknown column(s): {', '.join(unknown)}; allowed: {', '.join(COLUMNS)}")


def _read(handle: Iterable[str], max_rows: int) -> Loans:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise InputError("the file is empty")
    names = [n.strip() for n in reader.fieldnames]
    _check_header(names)
    reader.fieldnames = names
    return _build(((reader.line_num, rec) for rec in reader), max_rows)


def from_text(text: str, max_rows: int = MAX_ROWS) -> Loans:
    """Read a loan file given as CSV text. Row numbers in errors are line numbers."""
    try:
        return _read(text.splitlines(keepends=True), max_rows)
    except csv.Error as exc:
        raise InputError(f"not a readable CSV file ({exc})") from None


def read_csv(path: str | Path, max_rows: int = MAX_ROWS) -> Loans:
    """Read a loan file (UTF-8). Columns: loan_id, exposure, pd, lgd; optional borrower_id, sector, lgd_sd.

    Column names are exact and lower-case; unknown columns are refused rather than ignored, so a
    misspelt ``sector`` cannot silently turn a sector model into a one-sector run.
    """
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            return _read(handle, max_rows)
    except UnicodeDecodeError:
        raise InputError(f"{Path(path).name}: not a UTF-8 text file; save it as CSV UTF-8") from None
    except csv.Error as exc:
        raise InputError(f"{Path(path).name}: not a readable CSV file ({exc})") from None


def group(loans: Loans, sector_order: Sequence[str] | None = None) -> Borrowers:
    """One line per borrower. All loans of a borrower must share its pd and sector.

    Sectors are numbered alphabetically. ``sector_order`` puts the named sectors first, in that
    order, and any others after them.
    """
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
    present = set(sector_of)
    head = [s for s in (sector_order or ()) if s in present]
    names = tuple(head) + tuple(sorted(present - set(head)))
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
