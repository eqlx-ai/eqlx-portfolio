# Changelog

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
- Command line `eqlx-portfolio` and a read-only MCP server with 9 tools.
- Checked against the worked examples of the CreditMetrics Technical Document and of the
  CreditRisk+ document, closed forms,
  direct enumeration and brute-force simulation.
