import asyncio
import json
import os

import pytest

from eqlx_portfolio import cli


def run(capsys, *args):
    code = cli.main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def folder(tmp_path, capsys):
    assert run(capsys, "sample", tmp_path, "--borrowers", 120)[0] == 0
    settings = json.loads((tmp_path / "settings.json").read_text())
    settings["scenarios"] = 20_000
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    (tmp_path / "new.csv").write_text("loan_id,sector,exposure,pd,lgd\nNEW1,construction,30000000,0.02,0.45\n")
    return tmp_path


def test_sample_does_not_overwrite(folder, capsys):
    code, _, err = run(capsys, "sample", folder)
    assert code == 2 and "already exist" in err
    assert run(capsys, "sample", folder, "--force", "--borrowers", 20)[0] == 0
    assert run(capsys, "sample", folder, "--force", "--borrowers", 2)[0] == 2


def test_run_text_and_json(folder, capsys):
    loans, settings = folder / "loans.csv", folder / "settings.json"
    code, text, _ = run(capsys, "run", loans, "--config", settings)
    assert code == 0 and "economic capital" in text and "construction" in text
    code, out, _ = run(capsys, "run", loans, "--config", settings, "--json")
    res = json.loads(out)
    assert code == 0 and res["method"] == "simulation" and res["portfolio"]["borrowers"] == 120
    _, again, _ = run(capsys, "run", loans, "--config", settings, "--json")
    assert again == out
    crp_settings = folder / "crp.json"
    crp_settings.write_text(json.dumps({"method": "creditriskplus", "correlation": {"model": "single", "rho": 0.15}}))
    code, crp, _ = run(capsys, "run", loans, "--config", crp_settings, "--json", "--set", "confidence=[0.995]")
    assert code == 0 and json.loads(crp)["method"] == "creditriskplus"
    assert json.loads(crp)["levels"][0]["confidence"] == 0.995


def test_what_if_contributions_describe_pair(folder, capsys):
    loans, settings = folder / "loans.csv", folder / "settings.json"
    code, out, _ = run(capsys, "what-if", loans, "--config", settings, "--new", folder / "new.csv")
    assert code == 0 and json.loads(out)["levels"][-1]["expected_shortfall_change"] > 0
    code, out, _ = run(capsys, "contributions", loans, "--config", settings)
    lines = out.strip().splitlines()
    assert code == 0 and lines[0].startswith("borrower_id,sector,loans,exposure") and len(lines) == 121
    target = folder / "contrib.csv"
    assert run(capsys, "contributions", loans, "--config", settings, "--out", target)[0] == 0
    assert target.read_text().strip().splitlines() == lines
    code, out, _ = run(capsys, "describe", loans)
    assert code == 0 and json.loads(out)["borrowers"] == 120
    code, out, _ = run(capsys, "pair", "--pd-a", 0.0106, "--pd-b", 0.0006, "--rho", 0.2)
    d = json.loads(out)
    assert code == 0 and d["joint_default_probability"] > d["if_independent"]


def test_errors_exit_with_2(folder, capsys):
    loans, settings = folder / "loans.csv", folder / "settings.json"
    assert run(capsys, "run", str(loans) + ".missing", "--config", settings)[0] == 2
    code, _, err = run(capsys, "run", loans, "--config", settings, "--set", "mood=1")
    assert code == 2 and "unknown setting" in err
    assert run(capsys, "run", loans, "--config", settings, "--set", "seed")[0] == 2
    bad = folder / "bad.json"
    bad.write_text("{oops")
    assert run(capsys, "run", loans, "--config", bad)[0] == 2
    bad.write_text("[1]")
    assert run(capsys, "run", loans, "--config", bad)[0] == 2
    assert run(capsys, "pair", "--pd-a", 0.1, "--pd-b", 0.1, "--rho", 3)[0] == 2
    assert run(capsys, "mcp", "--root", folder / "nowhere")[0] == 2


mcp = pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from eqlx_portfolio.mcp_server import build_server  # noqa: E402


@pytest.fixture
def root(folder, tmp_path_factory):
    secret = tmp_path_factory.mktemp("outside") / "outside.csv"
    secret.write_text((folder / "loans.csv").read_text())
    os.symlink(secret, folder / "link_out.csv")
    (folder / "notes.txt").write_text("not a loan file")
    return folder


def call(server, name, args):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(name, args)

    return asyncio.run(go())


def data_of(result):
    assert not result.is_error, result.content
    return result.structured_content if result.structured_content is not None else json.loads(result.content[0].text)


TOOLS = {
    "list_files",
    "describe_loans",
    "portfolio_risk",
    "risk_contributions",
    "what_if_new_loans",
    "price_new_loan",
    "loan_price",
    "joint_default",
    "joint_migration",
    "exposure_value",
    "large_portfolio_benchmark",
}


def test_tools_are_read_only(root):
    async def go():
        async with Client(build_server(root)) as c:
            return await c.list_tools()

    tools = {t.name: t for t in asyncio.run(go()).tools}
    assert set(tools) == TOOLS
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools.values())
    assert all(t.description and len(t.description) > 40 for t in tools.values())


def test_mcp_flow(root):
    server = build_server(root)
    before = {p.name: p.stat().st_mtime_ns for p in root.iterdir()}
    files = data_of(call(server, "list_files", {}))
    files = files["result"] if isinstance(files, dict) else files
    assert {f["path"] for f in files} == {"loans.csv", "settings.json", "new.csv"}
    assert data_of(call(server, "describe_loans", {"loans": "loans.csv"}))["borrowers"] == 120
    args = {"loans": "loans.csv", "settings": "settings.json"}
    res = data_of(call(server, "portfolio_risk", args))
    assert res["levels"][-1]["economic_capital"] > 0 and len(res["top_contributors"]) == 10
    assert data_of(call(server, "portfolio_risk", args)) == res
    independent = {"correlation": {"model": "single", "rho": 0.0}}
    looser = data_of(call(server, "portfolio_risk", {**args, "overrides": independent}))
    assert looser["loss_sd"] < 0.95 * res["loss_sd"]
    contrib = data_of(call(server, "risk_contributions", {**args, "sector": "construction", "limit": 5}))
    assert len(contrib["borrowers"]) == 5 and all(b["sector"] == "construction" for b in contrib["borrowers"])
    new = [{"loan_id": "N1", "sector": "construction", "exposure": 30e6, "pd": 0.02, "lgd": 0.45}]
    w = data_of(call(server, "what_if_new_loans", {**args, "new_loans": new}))
    assert w["levels"][-1]["value_at_risk_before"] == res["levels"][-1]["value_at_risk"]
    assert w["levels"][-1]["expected_shortfall_change"] > 0
    assert {p.name: p.stat().st_mtime_ns for p in root.iterdir()} == before  # nothing was written


def test_mcp_calculators(root):
    server = build_server(root)
    j = data_of(call(server, "joint_default", {"pd_a": 0.02, "pd_b": 0.02, "asset_correlation": 0.3}))
    assert j["joint_default_probability"] > j["if_independent"] == pytest.approx(0.0004)
    m = data_of(
        call(server, "joint_migration", {"probs_a": [0.9, 0.1], "probs_b": [0.8, 0.2], "asset_correlation": 0.0})
    )
    assert m["joint_probabilities"][1][1] == pytest.approx(0.02)
    v = data_of(call(server, "exposure_value", {"probs": [0.9, 0.1], "values": [100, 40]}))
    assert v["mean"] == pytest.approx(94.0)
    b = data_of(call(server, "large_portfolio_benchmark", {"pd": 0.01, "lgd": 0.45, "asset_correlation": 0.2}))
    assert b["unexpected_loss_rate"] == pytest.approx(b["loss_rate_at_confidence"] - 0.0045)


@pytest.mark.parametrize("path", ["../outside.csv", "/etc/passwd", "link_out.csv", "notes.txt"])
def test_mcp_refuses_paths_outside_root_and_other_files(root, path):
    server = build_server(root)
    res = call(server, "describe_loans", {"loans": path})
    assert res.is_error and res.content[0].text != "Error executing tool describe_loans"


def test_mcp_reports_input_errors(root):
    server = build_server(root)
    args = {"loans": "loans.csv", "settings": "settings.json"}
    bad = call(server, "portfolio_risk", {**args, "overrides": {"mood": 1}})
    assert bad.is_error and "unknown setting" in bad.content[0].text
    assert call(server, "what_if_new_loans", {**args, "new_loans": []}).is_error
    assert call(server, "joint_default", {"pd_a": 0.02, "pd_b": 0.02, "asset_correlation": 2}).is_error
    assert call(server, "large_portfolio_benchmark", {"pd": 0.01, "lgd": 1.5, "asset_correlation": 0.2}).is_error


def test_mcp_contributions_filter_before_cutting_and_refuse_bad_arguments(root):
    server = build_server(root)
    args = {"loans": "loans.csv", "settings": "settings.json"}
    described = data_of(call(server, "describe_loans", {"loans": "loans.csv"}))
    in_sector = next(s["borrowers"] for s in described["concentration"]["by_sector"] if s["sector"] == "retail")
    res = data_of(call(server, "risk_contributions", {**args, "sector": "retail", "limit": 1000}))
    assert res["borrowers_matching"] == in_sector == len(res["borrowers"])
    for bad in ({"limit": 0}, {"limit": -5}, {"limit": 5000}, {"sector": "nosuch"}):
        out = call(server, "risk_contributions", {**args, **bad})
        assert out.is_error and out.content[0].text.startswith("Error executing tool risk_contributions: ")


def test_mcp_edge_values_are_json_safe_and_explained(root):
    server = build_server(root)
    m = data_of(
        call(server, "joint_migration", {"probs_a": [0.0, 0.9, 0.1], "probs_b": [0.8, 0.2], "asset_correlation": 0.3})
    )
    assert m["thresholds_a"][0] is None and m["joint_probabilities"][0] == [0.0, 0.0]
    j = data_of(call(server, "joint_default", {"pd_a": 0.0, "pd_b": 0.02, "asset_correlation": 0.3}))
    assert j["joint_default_probability"] == 0.0 and j["default_correlation"] is None
    for name, arguments in (
        ("exposure_value", {"probs": [0.5, 0.5], "values": [1e308, -1e308]}),
        ("describe_loans", {"loans": "a\x00b.csv"}),
        (
            "what_if_new_loans",
            {
                "loans": "loans.csv",
                "settings": "settings.json",
                "new_loans": [{"loan_id": "N", "exposure": 1, "pd": 0.1, "lgd": 0.5, "sektor": "x"}],
            },
        ),
    ):
        out = call(server, name, arguments)
        assert out.is_error and out.content[0].text.startswith(f"Error executing tool {name}: "), name
    (root / "latin.csv").write_bytes(b"loan_id,exposure,pd,lgd,sector\nL1,1000,0.02,0.45,in\xfeaat\n")
    out = call(server, "describe_loans", {"loans": "latin.csv"})
    assert out.is_error and "UTF-8" in out.content[0].text


TERMS = {"years": 3, "funding": 0.40, "target_return": 0.45, "capital_return": 0.38, "tax_rate": 0.30, "rate": 0.43}


def test_price_command(folder, capsys):
    loans, settings = folder / "loans.csv", folder / "settings.json"
    terms = folder / "terms.json"
    terms.write_text(json.dumps(TERMS))
    code, out, _ = run(capsys, "price", loans, "--config", settings, "--new", folder / "new.csv", "--terms", terms)
    d = json.loads(out)
    assert code == 0 and [v["capital_basis"] for v in d["views"]] == ["portfolio", "basel_formula"]
    assert d["views"][0]["hurdle_rate"] > 0.40
    terms.write_text(json.dumps({**TERMS, "mood": 1}))
    code, _, err = run(capsys, "price", loans, "--config", settings, "--new", folder / "new.csv", "--terms", terms)
    assert code == 2 and "unknown term" in err


def test_mcp_pricing_tools(root):
    server = build_server(root)
    new = {"loan_id": "N1", "sector": "construction", "exposure": 30e6, "pd": 0.02, "lgd": 0.45}
    args = {"loans": "loans.csv", "settings": "settings.json", "new_loan": new, "terms": TERMS}
    priced = data_of(call(server, "price_new_loan", args))
    view = priced["views"][0]
    assert view["capital_basis"] == "portfolio" and view["components"]["funding_cost_rate"] == pytest.approx(0.40)
    alone = data_of(
        call(
            server,
            "loan_price",
            {"notional": 30e6, "years": 3, "pd": 0.02, "lgd": 0.45, "capital": view["capital_per_unit_lent"],
             "target_return": 0.45, "funding": 0.40, "capital_return": 0.38, "tax_rate": 0.30, "rate": 0.43},
        )
    )  # fmt: skip
    assert alone["hurdle_rate"] == pytest.approx(view["hurdle_rate"]) and alone["raroc"] == pytest.approx(view["raroc"])
    bad = call(server, "price_new_loan", {**args, "terms": {"years": 3}})
    assert bad.is_error and "terms need" in bad.content[0].text
    bad = call(
        server,
        "loan_price",
        {"notional": 1, "years": 2.3, "pd": 0.02, "lgd": 0.45, "capital": 0.1, "target_return": 0.2, "funding": 0.1},
    )
    assert bad.is_error and "payment periods" in bad.content[0].text
