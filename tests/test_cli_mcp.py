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
