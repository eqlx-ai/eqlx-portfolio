"""The numbers printed in the README are real."""

import csv
import io
import json
import re
from pathlib import Path

import numpy as np

from eqlx_portfolio import (
    cli,
    config,
    creditriskplus,
    data,
    default_thresholds,
    engine,
    joint_default_probability,
    large_portfolio_loss,
    migration_thresholds,
    sigma_from_asset_correlation,
    simulate,
    single_factor,
    synthetic,
    value_stats,
)

README = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")


def _sample():
    loans_csv, settings_json = synthetic.generate()
    return data.from_records(csv.DictReader(io.StringIO(loans_csv))), config.from_dict(json.loads(settings_json))


def test_command_line_output_shown_in_the_readme():
    loans, settings = _sample()
    block = re.search(r"```text\n(eqlx-portfolio .*?)```", README, re.S)
    assert block is not None
    assert block.group(1).strip() == cli._text(engine.run(loans, settings)).strip()


def test_what_if_table():
    loans, settings = _sample()
    for sector, shortfall, per_unit in (("construction", "+8,530,008", "17.1%"), ("energy", "+3,824,232", "7.6%")):
        new = data.from_records([{"loan_id": "NEW1", "sector": sector, "exposure": 50e6, "pd": 0.02, "lgd": 0.45}])
        w = engine.what_if(loans, new, settings)
        assert f"{w['expected_loss_after'] - w['expected_loss_before']:+,.0f}" == "+450,000"
        assert f"{w['levels'][-1]['expected_shortfall_change']:+,.0f}" == shortfall
        assert f"{w['capital_per_unit_of_new_exposure']:.1%}" == per_unit
        assert f"| to {sector} | +450,000 | {shortfall} | {per_unit} |" in README


def test_library_example():
    bbb = [0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0117, 0.0012, 0.0018]
    values = [109.37, 109.19, 108.66, 107.55, 102.02, 98.10, 83.64, 51.13]
    s = value_stats(bbb, values)
    assert (round(s.mean, 2), round(s.sd, 2)) == (107.09, 2.99)
    assert str(np.round(migration_thresholds(bbb), 2)) == "[ 3.54  2.7   1.53 -1.49 -2.18 -2.75 -2.91]"
    joint = joint_default_probability(0.0018, 0.0006, 0.30)[0]
    assert f"{joint:.3e}" == "1.561e-05" and round(joint / (0.0018 * 0.0006), 1) == 14.5
    pd = np.full(100, 0.02)
    amount = np.full(100, 1_000_000 * 0.45)
    sim = simulate(
        single_factor(0.20, 100),
        default_thresholds(pd),
        np.column_stack((np.zeros(100), amount)),
        scenarios=200_000,
        seed=1,
        tail_levels=[0.999],
    )
    tail = sim.tails[0]
    assert (round(tail.value_at_risk), round(tail.expected_shortfall)) == (10800000, 13137313)
    assert round(large_portfolio_loss(amount, pd, 0.20, 0.999)) == 10184076
    sigma = sigma_from_asset_correlation(0.02, 0.20)
    res = creditriskplus(amount, pd, np.ones((100, 1)), [sigma], confidence=[0.999])
    assert round(sigma, 3) == 1.323
    assert (round(res.tails[0].value_at_risk), round(res.tails[0].expected_shortfall)) == (9900000, 11375006)
    for shown in ("(107.09, 2.99)", "1.561e-05", "(10800000, 13137313)", "10184076", "1.323", "(9900000, 11375006)"):
        assert shown in README


def test_tool_count_and_test_count_claims():
    assert README.count("| `") >= 9 + 7  # nine MCP tools and seven input columns are listed
    assert "9 tools" in (Path(__file__).parent.parent / "CHANGELOG.md").read_text(encoding="utf-8")


def test_one_version_everywhere():
    import eqlx_portfolio

    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{eqlx_portfolio.__version__}"' in pyproject
    assert f"## {eqlx_portfolio.__version__}" in (Path(__file__).parent.parent / "CHANGELOG.md").read_text()


def test_creditriskplus_figures_shown_in_the_readme():
    from . import published as pub

    pd = np.array([pub.CRP_DEFAULT_RATE[r] for r in pub.CRP_RATING])
    res = creditriskplus(pub.CRP_EXPOSURE, pd, np.ones((25, 1)), [0.5], loss_unit=10_000, confidence=[0.99, 0.999])
    shown = (f"{res.sd:,.0f}", f"{res.tails[0].value_at_risk:,.0f}", f"{res.tails[1].value_at_risk:,.0f}")
    assert shown == ("12,613,315", "55,240,000", "76,990,000")
    assert all(x in README for x in shown)
    assert f"{1 - res.sd / pub.CRP_SD:.1%}" == "0.4%"
    n_tests = int(re.search(r"(\d+) tests\.", README).group(1))
    assert n_tests > 100
