# eqlx-portfolio

Deterministic credit portfolio risk: how much a loan book can lose in a bad year, how much capital
that takes, and which borrowers and sectors drive it. A Python library, a command line tool and a
read-only MCP server for AI agents.

Loans are assessed one by one, but they fail together: a hundred loans to a hundred industries and a
hundred loans to one industry look the same loan by loan and are very different portfolios. This
tool measures that difference.

It computes. It does not estimate default probabilities, recoveries or correlations: those are
inputs. Same input, same output, and every figure comes with the steps that produced it.

- **Two methods, one input.** A seeded simulation of the asset value model (the approach of
  CreditMetrics) and the closed-form CreditRisk+ model. Use one to question the other.
- **Checked against published numbers.** The worked examples of the CreditMetrics Technical
  Document and of the CreditRisk+ document are reproduced in the test-suite; everything else is checked against closed forms, direct
  enumeration and brute-force simulation.
- **Built for agents.** The agent decides what to look at; the tool returns the number, the band
  around it and the steps taken. The MCP server only reads files in one folder and writes nothing.

## Who it is for

Credit risk and capital planning teams who want an independent view of economic capital and
concentration, credit committees asking what one more loan does to the book, validators who need a
transparent benchmark for a vendor model, and people building AI agents for credit work.

## Install

```bash
pip install eqlx-portfolio            # library and command line
pip install "eqlx-portfolio[mcp]"     # with the MCP server
```

Python 3.10 or newer. Depends on NumPy and SciPy only.

## A first look

```bash
eqlx-portfolio sample demo            # writes a made-up portfolio: demo/loans.csv, demo/settings.json
eqlx-portfolio run demo/loans.csv --config demo/settings.json
```

```text
eqlx-portfolio 0.1.0  method: simulation
loans 621  borrowers 500  sectors 6  exposure 995,349,000

expected loss                12,716,172
loss std deviation           12,031,533
at 99.00% confidence
  value-at-risk              57,233,350   95% band 56,635,700 to 57,801,100
  expected shortfall         71,379,023
  economic capital           44,517,178
at 99.90% confidence
  value-at-risk              91,343,950   95% band 88,673,050 to 93,627,200
  expected shortfall        104,348,278
  economic capital           78,627,778

largest borrower 5.5% of exposure, ten largest 28.0%, effective number of borrowers 79

sector                exposure share  risk share
construction                   41.1%       69.1%
services                       19.4%       13.0%
manufacturing                  14.8%        6.8%
retail                         11.7%        4.7%
tourism                         5.6%        3.5%
energy                          7.4%        2.9%

borrower              exposure share  risk share
B0004                           4.0%       12.7%
B0003                           4.7%       10.6%
B0005                           4.2%        8.8%
B0001                           5.5%        7.4%
B0002                           2.4%        3.8%
B0287                           1.8%        3.6%
B0303                           1.1%        2.7%
B0136                           0.7%        2.6%
B0407                           1.0%        2.1%
B0435                           0.5%        2.0%
```

Construction is 41% of the exposure and 69% of the risk: five large borrowers were planted there.
The band next to each value-at-risk says how far simulation noise can move it. The same run with
`--json` adds the checks that were made and the steps taken.

What does one more loan do? Fifty million to a new construction borrower, then the same loan to an
energy borrower:

```bash
eqlx-portfolio what-if demo/loans.csv --config demo/settings.json --new new_loan.csv
```

| New loan of 50,000,000 (pd 2%, lgd 45%) | Expected loss | Expected shortfall at 99.9% | Capital per unit lent |
|---|---|---|---|
| to construction | +450,000 | +8,530,008 | 17.1% |
| to energy | +450,000 | +3,824,232 | 7.6% |

Same loan, same expected loss, twice the capital: it lands where the book is already heavy. Both
portfolios are valued on the same scenarios, so the difference is not simulation noise.

Other commands: `contributions` (every borrower's share of the risk, as CSV), `describe` (what a
loan file contains), `pair` (joint default probability of two borrowers).

## As a library

```python
import numpy as np
from eqlx_portfolio import (
    creditriskplus,
    default_thresholds,
    joint_default_probability,
    large_portfolio_loss,
    migration_thresholds,
    sigma_from_asset_correlation,
    simulate,
    single_factor,
    value_stats,
)

# One bond, rated BBB: where can it be in a year, and what is it worth there?
bbb = [0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0117, 0.0012, 0.0018]  # AAA ... CCC, default
values = [109.37, 109.19, 108.66, 107.55, 102.02, 98.10, 83.64, 51.13]
s = value_stats(bbb, values)
round(s.mean, 2), round(s.sd, 2)  # (107.09, 2.99)
np.round(migration_thresholds(bbb), 2)  # [ 3.54  2.7   1.53 -1.49 -2.18 -2.75 -2.91]

# Two borrowers with 0.18% and 0.06% default probability and 30% asset correlation
joint_default_probability(0.0018, 0.0006, 0.30)[0]  # 1.561e-05, 14.5 times the independent 1.08e-06

# 100 loans of 1,000,000, pd 2%, lgd 45%, one common factor with 20% asset correlation
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
round(tail.value_at_risk), round(tail.expected_shortfall)  # (10800000, 13137313)
round(large_portfolio_loss(amount, pd, 0.20, 0.999))  # 10184076: the infinitely fine limit

# The same portfolio in CreditRisk+, with the sector volatility that matches the 20% correlation
sigma = sigma_from_asset_correlation(0.02, 0.20)  # 1.323
res = creditriskplus(amount, pd, np.ones((100, 1)), [sigma], confidence=[0.999])
round(res.tails[0].value_at_risk), round(res.tails[0].expected_shortfall)  # (9900000, 11375006)
```

## For AI agents (MCP)

```bash
eqlx-portfolio mcp --root /path/to/folder
```

Client configuration (Claude Desktop, Claude Code and other MCP clients):

```json
{"mcpServers": {"eqlx-portfolio": {"command": "eqlx-portfolio", "args": ["mcp", "--root", "/path/to/folder"]}}}
```

| Tool | What it returns |
|---|---|
| `list_files` | loan and settings files in the folder |
| `describe_loans` | loans, borrowers, sectors, exposure, expected loss, largest borrowers |
| `portfolio_risk` | expected loss, value-at-risk, expected shortfall, economic capital, concentration, steps |
| `risk_contributions` | borrowers ranked by their share of the risk, next to their share of exposure |
| `what_if_new_loans` | the portfolio before and after new loans, on the same scenarios |
| `joint_default` | probability that two borrowers default together |
| `joint_migration` | joint year-end rating probabilities of two borrowers |
| `exposure_value` | mean and standard deviation of one exposure's value |
| `large_portfolio_benchmark` | loss rate of a perfectly fine-grained portfolio in a bad year |

The server reads `.csv` and `.json` files inside `--root` and nothing else; `..` and symlinks cannot
leave the folder. It writes nothing and sends nothing anywhere. When an input is refused the agent
gets the reason, not a bare error.

## Input

Loan file (CSV), one row per loan:

| Column | Meaning |
|---|---|
| `loan_id` | unique |
| `exposure` | amount at risk if the borrower defaults |
| `pd` | default probability over the horizon, at least 0 and below 1 |
| `lgd` | share of the exposure lost in default, 0 to 1 |
| `borrower_id` | optional; loans of one borrower default together |
| `sector` | optional; needed for the sector correlation model |
| `lgd_sd` | optional; makes the loss in default uncertain (simulation only) |

Settings (JSON). The correlation block is required: it is the main assumption of a run.

```json
{
  "method": "simulation",
  "confidence": [0.99, 0.999],
  "scenarios": 200000,
  "seed": 1,
  "correlation": {
    "model": "sector",
    "within": {"construction": 0.24, "energy": 0.18},
    "default_within": 0.15,
    "default_between": 0.5,
    "between": {"construction|energy": 0.3}
  }
}
```

| Correlation model | Meaning |
|---|---|
| `single` | one common factor; `rho` is the asset correlation of any two borrowers |
| `basel_corporate` | one common factor; each borrower's correlation from the Basel corporate formula (12% to 24% by pd) |
| `sector` | one factor per sector; `within` a sector, and `between` the sector factors |

`"method": "creditriskplus"` solves the portfolio in closed form instead. It accepts the `single`
and `sector` models with independent sectors; the sector volatility is derived from the asset
correlation unless given in `"creditriskplus": {"sector_sigma": {...}}`.

## How it is verified

Published figures, from the CreditMetrics Technical Document (J.P. Morgan, 1997):

| Example | Published | Reproduced |
|---|---|---|
| Value of a BBB bond after one year: mean, standard deviation (Table 2.5) | 107.09, 2.99 | 107.09, 2.99 |
| The same with uncertain recovery | 3.18 | 3.18 |
| Revaluation of the bond in each rating from the forward curves (Tables 1.2, 1.4) | 16 values | within 0.02 |
| Asset return thresholds of BB and single-A borrowers (Tables 8.5, 8.6) | 14 values | all 14 |
| Joint migration of two borrowers, 0.30 asset correlation (Table 3.2) | 64 cells | within 0.01 point |
| Joint migration of two borrowers, 0.20 asset correlation (Table 8.7) | 64 cells | within 0.05 point |
| Scenarios of asset returns mapped to ratings (Tables 10.4, 10.5) | 30 ratings | all 30 |
| First percentile of a two-bond portfolio | 204.40 | 204.40 |
| Three-bond portfolio: means and variances (chapter 9) | 7 values | all 7 |

Published figures, from the CreditRisk+ document (Credit Suisse First Boston, 1997), example 1A,
25 obligors in one sector:

| Figure | Published | Reproduced |
|---|---|---|
| Aggregate exposure, expected loss | 130,513,072 and 14,221,863 | the same |
| Standard deviation | 12,668,742 | 12,613,315 (0.4% lower) |
| 99th percentile loss | 55,311,503 | 55,240,000 (0.1% lower) |
| 99.9th percentile loss | 77,133,478 | 76,990,000 (0.2% lower) |

The document's spreadsheet groups exposures into bands by a rounding rule it does not spell out;
the small gaps are that rounding.

Three figures printed in the CreditMetrics document do not follow from its own tables (a misprinted probability, a
misprinted threshold, and covariances that the model cannot produce). The tests show where each
comes from; see [LIMITS.md](LIMITS.md).

Independent truth for everything else:

- the bivariate normal distribution against SciPy, to 12 decimals;
- simulated mean and standard deviation against exact pairwise formulas;
- a fine-grained portfolio against the large-portfolio formula behind the Basel risk weights;
- CreditRisk+ against the negative binomial and Poisson closed forms (to 14 decimals) and against a
  brute-force simulation of the same model, including every borrower's risk contribution;
- risk contributions add up to the expected shortfall, in both methods;
- a what-if run reproduces, to the last digit, the separate run of the portfolio without the new
  loans, also when the new loan brings a new sector or an uncertain loss given default;
- every defect found by an independent review is pinned by a regression test.

145 tests. `uv run pytest` runs them in about a minute.

## Limits

One period, default mode in portfolio runs, inputs are yours, the far tail of a simulation is thin,
CreditRisk+ sectors are independent. No estimation of correlations, no granularity adjustment, no
stress scenarios, not regulatory capital. Details in [LIMITS.md](LIMITS.md).

## References

- Gupton, G. M., Finger, C. C., Bhatia, M. (1997). *CreditMetrics - Technical Document*. J.P. Morgan.
- Credit Suisse Financial Products (1997). *CreditRisk+: A Credit Risk Management Framework*.
- Vasicek, O. (2002). The distribution of loan portfolio value. *Risk*, December.
- Gordy, M. B. (2000). A comparative anatomy of credit risk models. *Journal of Banking & Finance* 24.
- Gordy, M. B. (2003). A risk-factor model foundation for ratings-based bank capital rules.
  *Journal of Financial Intermediation* 12.
- Haaf, H., Tasche, D. (2002). Calculating value-at-risk contributions in CreditRisk+. *GARP Risk Review* 7.
- Haaf, H., Reiss, O., Schoenmakers, J. (2004). Numerically stable computation of CreditRisk+.
  In *CreditRisk+ in the Banking Industry*, Springer.
- Basel Committee on Banking Supervision (2005). *An Explanatory Note on the Basel II IRB Risk Weight
  Functions*.
- Drezner, Z., Wesolowsky, G. O. (1990). On the computation of the bivariate normal integral.
  *Journal of Statistical Computation and Simulation* 35.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
