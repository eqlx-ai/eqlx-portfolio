"""A made-up loan portfolio for trying the tool. No real borrower, bank or figure is behind it."""

from __future__ import annotations

import io
import json

import numpy as np

SECTORS = ("construction", "energy", "manufacturing", "retail", "services", "tourism")
_WEIGHTS = (0.22, 0.10, 0.25, 0.18, 0.17, 0.08)
_WITHIN = {
    "construction": 0.24,
    "energy": 0.18,
    "manufacturing": 0.16,
    "retail": 0.14,
    "services": 0.12,
    "tourism": 0.22,
}
_GRADES = ((0.003, 0.10), (0.008, 0.25), (0.02, 0.35), (0.05, 0.20), (0.12, 0.10))


def generate(borrowers: int = 500, seed: int = 7) -> tuple[str, str]:
    """Loan file (CSV text) and settings (JSON text). A few large borrowers are planted in construction."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out = io.StringIO()
    out.write("loan_id,borrower_id,sector,exposure,pd,lgd\n")
    loan = 0
    for i in range(1, borrowers + 1):
        sector = SECTORS[int(rng.choice(len(SECTORS), p=_WEIGHTS))]
        grade = int(rng.choice(len(_GRADES), p=[g[1] for g in _GRADES]))
        pd = _GRADES[grade][0]
        size = float(np.round(rng.lognormal(mean=14.0, sigma=1.0), -3))
        if i <= 5:  # planted concentration: five large construction names
            sector, size, pd = "construction", float(40_000_000 + 5_000_000 * i), 0.02
        for _ in range(1 + int(rng.random() < 0.25)):
            loan += 1
            lgd = float(rng.choice((0.25, 0.40, 0.45, 0.60)))
            part = max(1000.0, float(np.round(size * rng.uniform(0.3, 1.0), -3)))
            out.write(f"L{loan:05d},B{i:04d},{sector},{part:.0f},{pd},{lgd}\n")
    settings = {
        "method": "simulation",
        "confidence": [0.99, 0.999],
        "scenarios": 200000,
        "seed": 1,
        "correlation": {"model": "sector", "within": _WITHIN, "default_between": 0.5},
        "top": 10,
    }
    return out.getvalue(), json.dumps(settings, indent=2) + "\n"
