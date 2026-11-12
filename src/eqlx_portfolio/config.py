"""Settings of a portfolio run. Unknown keys are refused rather than ignored."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._checks import Array
from .errors import InputError

METHODS = ("simulation", "creditriskplus")
CORRELATION_MODELS = ("single", "basel_corporate", "sector")
_TOP = {"method", "confidence", "scenarios", "seed", "correlation", "creditriskplus", "top"}
_CORR = {"model", "rho", "within", "default_within", "between", "default_between"}
_CRP = {"loss_unit", "sector_sigma", "default_sigma"}


@dataclass(frozen=True)
class Correlation:
    model: str
    rho: float | None
    within: Mapping[str, float]
    default_within: float | None
    between: Mapping[tuple[str, str], float]
    default_between: float


@dataclass(frozen=True)
class Settings:
    method: str
    confidence: tuple[float, ...]
    scenarios: int
    seed: int
    correlation: Correlation
    loss_unit: float | None
    sector_sigma: Mapping[str, float]
    default_sigma: float | None
    top: int

    def to_dict(self) -> dict[str, Any]:
        c = self.correlation
        corr: dict[str, Any] = {"model": c.model}
        if c.model == "single":
            corr["rho"] = c.rho
        if c.model == "sector":
            corr["within"] = dict(c.within)
            corr["default_within"] = c.default_within
            corr["between"] = {f"{a}|{b}": v for (a, b), v in c.between.items()}
            corr["default_between"] = c.default_between
        out: dict[str, Any] = {"method": self.method, "confidence": list(self.confidence), "correlation": corr}
        if self.method == "simulation":
            out["scenarios"] = self.scenarios
            out["seed"] = self.seed
        else:
            out["creditriskplus"] = {
                "loss_unit": self.loss_unit,
                "sector_sigma": dict(self.sector_sigma),
                "default_sigma": self.default_sigma,
            }
        out["top"] = self.top
        return out


def _unknown(where: str, raw: Mapping[str, Any], allowed: set[str]) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        raise InputError(f"unknown setting(s) in {where}: {', '.join(extra)}")


def _real(name: str, value: Any, low: float, high: float, *, open_low: bool = False, open_high: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a number")
    try:
        v = float(value)
    except OverflowError:
        raise InputError(f"{name} is outside its allowed range") from None
    if not np.isfinite(v):
        raise InputError(f"{name} must be a number")
    if v < low or v > high or (open_low and v == low) or (open_high and v == high):
        raise InputError(f"{name} is outside its allowed range")
    return v


def _whole(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise InputError(f"{name} must be a whole number between {low} and {high}")
    return value


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputError(f"{name} must be an object")
    return value


def read_json(path: Any) -> dict[str, Any]:
    """A settings file: UTF-8 JSON holding one object."""
    import json
    from pathlib import Path

    name = Path(path).name
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError:
        raise InputError(f"{name}: not a UTF-8 text file") from None
    except (json.JSONDecodeError, RecursionError) as exc:
        raise InputError(f"{name}: not valid JSON ({exc})") from None
    if not isinstance(raw, dict):
        raise InputError(f"{name}: the settings must be a JSON object")
    return raw


def from_dict(raw: Mapping[str, Any]) -> Settings:
    raw = _mapping("settings", raw)
    _unknown("settings", raw, _TOP)
    method = raw.get("method", "simulation")
    if method not in METHODS:
        raise InputError(f"method must be one of: {', '.join(METHODS)}")
    conf = raw.get("confidence", [0.99, 0.999])
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        conf = [conf]
    if not isinstance(conf, list) or not conf:
        raise InputError("confidence must be a number or a list of numbers")
    levels = tuple(sorted({_real("confidence", c, 0.5, 1.0, open_high=True) for c in conf}))

    if "correlation" not in raw:
        raise InputError("settings need a correlation block: it is the main assumption of the run")
    c = _mapping("correlation", raw["correlation"])
    _unknown("correlation", c, _CORR)
    model = c.get("model")
    if model not in CORRELATION_MODELS:
        raise InputError(f"correlation.model must be one of: {', '.join(CORRELATION_MODELS)}")
    rho = None
    within: dict[str, float] = {}
    default_within = None
    between: dict[tuple[str, str], float] = {}
    default_between = 0.0
    if model == "single":
        if "rho" not in c:
            raise InputError("correlation.rho is required for the single model")
        rho = _real("correlation.rho", c["rho"], 0.0, 1.0, open_high=True)
    if model == "sector":
        for k, v in _mapping("correlation.within", c.get("within", {})).items():
            within[str(k)] = _real(f"correlation.within[{k}]", v, 0.0, 1.0, open_high=True)
        if c.get("default_within") is not None:
            default_within = _real("correlation.default_within", c["default_within"], 0.0, 1.0, open_high=True)
        if not within and default_within is None:
            raise InputError("the sector model needs correlation.within or correlation.default_within")
        for k, v in _mapping("correlation.between", c.get("between", {})).items():
            parts = [p.strip() for p in str(k).split("|")]
            if len(parts) != 2 or parts[0] == parts[1]:
                raise InputError(f"correlation.between key {k!r} must name two different sectors as 'a|b'")
            a, b = sorted(parts)
            if (a, b) in between:
                raise InputError(f"correlation.between names the pair {a}|{b} twice")
            between[(a, b)] = _real(f"correlation.between[{k}]", v, -1.0, 1.0)
        default_between = _real("correlation.default_between", c.get("default_between", 0.0), -1.0, 1.0)
    extra = (
        set(c)
        - {"model"}
        - ({"rho"} if model == "single" else set())
        - (_CORR - {"rho", "model"} if model == "sector" else set())
    )
    if extra:
        raise InputError(f"correlation.{sorted(extra)[0]} does not apply to the {model} model")

    crp = _mapping("creditriskplus", raw.get("creditriskplus", {}))
    _unknown("creditriskplus", crp, _CRP)
    if crp and method != "creditriskplus":
        raise InputError("the creditriskplus block applies only when method is creditriskplus")
    loss_unit = None
    if crp.get("loss_unit") is not None:
        loss_unit = _real(
            "creditriskplus.loss_unit", crp["loss_unit"], 0.0, float("inf"), open_low=True, open_high=True
        )
    sigma = {
        str(k): _real(f"creditriskplus.sector_sigma[{k}]", v, 0.0, 10.0)
        for k, v in _mapping("creditriskplus.sector_sigma", crp.get("sector_sigma", {})).items()
    }
    default_sigma = None
    if crp.get("default_sigma") is not None:
        default_sigma = _real("creditriskplus.default_sigma", crp["default_sigma"], 0.0, 10.0)
    if method == "creditriskplus":
        if model == "basel_corporate":
            raise InputError("CreditRisk+ needs the single or the sector correlation model")
        if between or default_between != 0.0:
            raise InputError(
                "CreditRisk+ treats sectors as independent: remove between-sector correlations or use simulation"
            )
        if "scenarios" in raw or "seed" in raw:
            raise InputError("scenarios and seed apply only to simulation")

    return Settings(
        method=method,
        confidence=levels,
        scenarios=_whole("scenarios", raw.get("scenarios", 100_000), 1_000, 20_000_000),
        seed=_whole("seed", raw.get("seed", 1), 0, 2**63 - 1),
        correlation=Correlation(model, rho, within, default_within, between, default_between),
        loss_unit=loss_unit,
        sector_sigma=sigma,
        default_sigma=default_sigma,
        top=_whole("top", raw.get("top", 10), 0, 2_000_000),
    )


def with_overrides(raw: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Replace single settings by dotted key, e.g. ``{"correlation.rho": 0.25}``."""
    out: dict[str, Any] = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in raw.items()}
    for key, value in overrides.items():
        parts = str(key).split(".")
        node = out
        for part in parts[:-1]:
            nxt = node.get(part)
            if nxt is None:
                nxt = {}
            if not isinstance(nxt, dict):
                raise InputError(f"cannot set {key}: {part} is not an object")
            node[part] = dict(nxt)
            node = node[part]
        node[parts[-1]] = value
    return out


def sector_within(settings: Settings, sector_names: tuple[str, ...]) -> Array:
    c = settings.correlation
    unknown = sorted(set(c.within) - set(sector_names))
    if unknown:
        raise InputError(f"correlation.within names sectors that are not in the loan file: {', '.join(unknown)}")
    out = []
    for s in sector_names:
        if s in c.within:
            out.append(c.within[s])
        elif c.default_within is not None:
            out.append(c.default_within)
        else:
            raise InputError(f"no within-sector correlation for sector {s!r} (set default_within)")
    return np.array(out)


def sector_between(settings: Settings, sector_names: tuple[str, ...]) -> Array:
    c = settings.correlation
    k = len(sector_names)
    m = np.full((k, k), c.default_between)
    np.fill_diagonal(m, 1.0)
    pos = {s: i for i, s in enumerate(sector_names)}
    for (a, b), v in c.between.items():
        if a not in pos or b not in pos:
            raise InputError(f"correlation.between names a sector that is not in the loan file: {a}|{b}")
        m[pos[a], pos[b]] = m[pos[b], pos[a]] = v
    return m
