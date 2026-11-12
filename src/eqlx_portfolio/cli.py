"""Command line: ``eqlx-portfolio run | what-if | price | contributions | describe | pair | sample | mcp``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from . import config, data, engine, synthetic
from ._version import __version__
from .errors import InputError


def _settings(path: str, overrides: list[str] | None) -> config.Settings:
    raw = config.read_json(path)
    changes: dict[str, Any] = {}
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise InputError(f"--set expects key=value, got {item!r}")
        try:
            changes[key] = json.loads(value)
        except (json.JSONDecodeError, RecursionError):
            changes[key] = value
    return config.from_dict(config.with_overrides(raw, changes) if changes else raw)


def _emit(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _money(x: float) -> str:
    return f"{x:,.0f}"


def _text(res: dict[str, Any]) -> str:
    p = res["portfolio"]
    lines = [
        f"eqlx-portfolio {res['tool']['version']}  method: {res['method']}",
        f"loans {p['loans']}  borrowers {p['borrowers']}  sectors {p['sectors']}  exposure {_money(p['exposure'])}",
        "",
        f"expected loss        {_money(res['expected_loss']):>18}",
        f"loss std deviation   {_money(res['loss_sd']):>18}",
    ]
    for lv in res["levels"]:
        band = lv.get("value_at_risk_95_band")
        shown = ""
        if band:
            shown = f"   95% band {_money(band[0])} to {_money(band[1]) if band[1] is not None else 'unbounded'}"
        lines += [
            f"at {lv['confidence']:.2%} confidence",
            f"  value-at-risk      {_money(lv['value_at_risk']):>18}{shown}",
            f"  expected shortfall {_money(lv['expected_shortfall']):>18}",
            f"  economic capital   {_money(lv['economic_capital']):>18}",
        ]
    lines += [f"! {step}" for step in res["steps"] if step.startswith(("Warning:", "Note:"))]
    c = res["concentration"]
    lines += [
        "",
        f"largest borrower {c['largest_borrower_share']:.1%} of exposure, ten largest {c['ten_largest_share']:.1%}, "
        f"effective number of borrowers {c['effective_number_of_borrowers']:.0f}",
        "",
        f"{'sector':<20}{'exposure share':>16}{'risk share':>12}",
    ]
    lines += [f"{r['sector']:<20}{r['exposure_share']:>16.1%}{r['risk_share']:>12.1%}" for r in c["by_sector"]]
    if res["top_contributors"]:
        lines += ["", f"{'borrower':<20}{'exposure share':>16}{'risk share':>12}"]
        lines += [
            f"{r['borrower_id']:<20}{r['exposure_share']:>16.1%}{r['risk_share']:>12.1%}"
            for r in res["top_contributors"]
        ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eqlx-portfolio", description="Deterministic credit portfolio risk.")
    p.add_argument("--version", action="version", version=f"eqlx-portfolio {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def run_args(s: argparse.ArgumentParser) -> None:
        s.add_argument("loans", help="loan file (CSV)")
        s.add_argument("--config", required=True, help="settings file (JSON)")
        s.add_argument(
            "--set", action="append", metavar="KEY=VALUE", help="replace one setting, e.g. correlation.rho=0.2"
        )

    s = sub.add_parser("run", help="loss distribution, capital and risk shares of a portfolio")
    run_args(s)
    s.add_argument("--json", action="store_true", help="print the full result as JSON")

    s = sub.add_parser("what-if", help="what adding new loans does to the portfolio")
    run_args(s)
    s.add_argument("--new", required=True, help="loan file (CSV) with the loans to add")

    s = sub.add_parser("price", help="what a new loan has to earn: hurdle rate and RAROC")
    run_args(s)
    s.add_argument("--new", required=True, help="loan file (CSV) with the one loan to price")
    s.add_argument("--terms", required=True, help="pricing terms (JSON): years, funding, target_return, ...")

    s = sub.add_parser("contributions", help="every borrower's risk contribution as CSV")
    run_args(s)
    s.add_argument("--out", help="write to this file instead of standard output")

    s = sub.add_parser("describe", help="what a loan file contains, before any calculation")
    s.add_argument("loans")

    s = sub.add_parser("pair", help="joint default probability of two borrowers")
    s.add_argument("--pd-a", type=float, required=True)
    s.add_argument("--pd-b", type=float, required=True)
    s.add_argument("--rho", type=float, required=True, help="asset correlation")

    s = sub.add_parser("sample", help="write a made-up portfolio and settings to try the tool")
    s.add_argument("directory")
    s.add_argument("--borrowers", type=int, default=500)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--force", action="store_true", help="overwrite existing files")

    s = sub.add_parser("mcp", help="start the read-only MCP server for AI agents")
    s.add_argument("--root", required=True, help="the only folder the server may read")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (InputError, FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    cmd = args.command
    if cmd == "mcp":
        from .mcp_server import build_server

        build_server(Path(args.root)).run()
        return 0
    if cmd == "describe":
        _emit(engine.describe(data.read_csv(args.loans)))
        return 0
    if cmd == "pair":
        _emit(engine.pair(args.pd_a, args.pd_b, args.rho))
        return 0
    if cmd == "sample":
        if not 10 <= args.borrowers <= 100_000:
            raise InputError("--borrowers must be between 10 and 100000")
        out = Path(args.directory)
        out.mkdir(parents=True, exist_ok=True)
        existing = [f for f in ("loans.csv", "settings.json") if (out / f).exists()]
        if existing and not args.force:
            raise InputError(f"{', '.join(existing)} already exist in {out}; use --force to overwrite")
        loans_csv, settings_json = synthetic.generate(args.borrowers, args.seed)
        (out / "loans.csv").write_text(loans_csv, encoding="utf-8")
        (out / "settings.json").write_text(settings_json, encoding="utf-8")
        print(f"wrote loans.csv and settings.json to {out}")
        return 0
    loans = data.read_csv(args.loans)
    settings = _settings(args.config, args.set)
    if cmd == "run":
        res = engine.run(loans, settings)
        if args.json:
            _emit(res)
        else:
            print(_text(res))
        return 0
    if cmd == "what-if":
        _emit(engine.what_if(loans, data.read_csv(args.new), settings))
        return 0
    if cmd == "price":
        _emit(engine.price_new_loan(loans, data.read_csv(args.new), settings, config.read_json(args.terms)))
        return 0
    rows = engine.contributions(loans, settings)
    handle = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout  # noqa: SIM115
    try:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
