# Limits

What the tool does not do, and where its numbers need care.

## Inputs are yours

- **It estimates nothing.** Default probabilities, loss given default, exposures and correlations are
  inputs. The result is only as good as they are, and the correlation assumption moves the tail more
  than anything else. Run more than one.
- **One period.** The horizon is the period the default probabilities refer to, normally one year.
  There is no multi-year simulation.
- **Defaulted loans stay out.** A loan with a default probability of 1 is a loss already, not a risk.

## Default mode in portfolio runs

- A portfolio run (`run`, `what-if`, the MCP file tools) counts a loss only when a borrower defaults.
  Loss of value from a rating downgrade is available in the library (`bond_values_at_horizon`,
  `value_stats`, `portfolio_value_stats`, `simulate` with several states) but not from a loan file.
- Loss given default is fixed, or beta-distributed around its mean when `lgd_sd` is given. It does
  not move with the cycle: there is no link between default rates and recoveries.
- All loans of a borrower default together and must carry the same default probability and sector.
  Groups of connected borrowers are not merged: give them one `borrower_id` if you treat them as one.

## Simulation

- **Same input, same seed, same result** on the same machine and library versions. The random
  numbers themselves are identical everywhere (NumPy PCG64); across machines a result can differ only
  if a rounding difference in the last digit moves an asset return across a default threshold, which
  changes single scenarios, not the picture. NumPy's version is printed in every result.
- **The far tail is thin.** With 100,000 scenarios the 99.9% value-at-risk rests on 100 outcomes. The
  result prints a 95% band around every simulated value-at-risk; if it is too wide, raise `scenarios`.
- **Risk contributions are noisier than totals.** A borrower's share is its average loss over the
  tail outcomes; small borrowers default in few of them. Use more scenarios, or CreditRisk+, before
  ranking borrowers that are close together.
- Speed: about 5 seconds for 500 borrowers and 200,000 scenarios on a laptop; time grows with
  borrowers times scenarios. The exact standard deviation check runs up to 1,500 borrowers.

## CreditRisk+

- Sectors are independent of each other. A portfolio whose sectors move together needs the
  simulation, or one common sector.
- Defaults are treated as Poisson events, which is accurate for small default probabilities and
  overstates losses a little when they are large (above roughly 10%).
- Losses are rounded to whole loss units; each borrower's expected loss is kept. The unit is printed.
- It is a different model from the simulation, not a faster copy: given the same default
  probabilities and default correlation the two agree on expected loss and are close on the standard
  deviation, and drift apart in the far tail (gamma versus normal factors).

## Not included

- Estimation of correlations or sector volatilities from data.
- Granularity adjustment, stress scenarios, multi-period migration, mark-to-market of loans from a
  loan file, derivative exposure, commitments and drawdown behaviour.
- Regulatory capital. The large-portfolio benchmark is the formula behind the Basel risk weights
  without the maturity adjustment and the supervisory parameters; it is a yardstick, not a filing.
- Any statement about a real bank, borrower or portfolio. The sample portfolio is made up.

## Published examples that do not reproduce

The CreditRisk+ document's 25-obligor example is matched exactly on exposure and expected loss, and
to within 0.5% on the standard deviation and 0.2% on the tail percentiles: its spreadsheet groups
exposures into bands by a rounding rule the document does not spell out.

The CreditMetrics Technical Document (1997) is the source of most test values. Three of its figures
cannot be reproduced from its own tables, and the test-suite pins down why instead of matching them:

- Single-A bond mean 106.55 and standard deviation 1.49, and therefore the two-bond mean 213.63:
  Table 1.6 prints 0.60% for the move from A to B where the transition matrix has 0.26%. With 0.60%
  the probabilities add up to 100.34% and give exactly the printed figures. With the matrix the
  values are 106.20, 1.42 and 213.29.
- Table 10.2 prints 2.78 for one threshold of the BBB firm; its probabilities give 2.70.
- Chapter 9 prints covariances between the three bonds that imply a correlation of values above the
  asset correlation they come from. The pairwise formula is checked against simulation instead.
