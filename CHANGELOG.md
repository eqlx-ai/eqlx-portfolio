# Changelog

## 0.2.0

- Risk-based loan pricing: cost components (funding, expected loss, operating cost, capital), the
  hurdle rate at which a loan earns the target return, RAROC and yearly value added at an offered
  rate, for fixed-rate bullet and equal-principal loans on a funding rate or curve, with a default
  probability term structure and collateral.
- `price_new_loan`: the capital a new loan ties up is measured from the portfolio itself (it costs
  more where the book is already heavy), from the formula behind the Basel risk weights and from a
  fixed risk weight, and the loan is priced under each.
- Command `eqlx-portfolio price`; MCP tools `price_new_loan` and `loan_price` (11 tools in all).
- Checked against the closed form of the scheme it follows (Engelmann and Pham 2020) and hand
  arithmetic; the article's numerical example is reproduced approximately only (see LIMITS.md).

## 0.1.0

First release.

- Asset value model: migration thresholds, joint migration probabilities of two borrowers, joint
  default probability and default correlation from asset correlation.
- Value of one exposure at the horizon by year-end rating, its mean and standard deviation, with
  uncertain recovery; exact mean and standard deviation of portfolio value pair by pair, marginal
  standard deviations.
- Seeded Monte Carlo simulation with one common factor, the Basel corporate correlation or sector
  factors: loss distribution, value-at-risk with a distribution-free band, expected shortfall,
  economic capital, and each borrower's share of the expected shortfall.
- CreditRisk+ in closed form with a numerically stable expansion, sector analysis and risk
  contributions.
- Large-portfolio benchmark (the one-factor formula behind the Basel risk weights).
- Portfolio runs from a loan file (CSV) and settings (JSON): concentration by borrower and sector,
  top contributors, what-if for new loans on the same scenarios.
- Command line `eqlx-portfolio` and a read-only MCP server.
- Checked against the worked examples of the CreditMetrics Technical Document and of the
  CreditRisk+ document, closed forms,
  direct enumeration and brute-force simulation.
