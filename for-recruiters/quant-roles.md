# If you are hiring for a quant role

A two-minute tour of the parts of this repository that demonstrate
quantitative-research ability.

## What to look at first

1. **`docs/methodology.md`**, the rigor showcase. Walk-forward validation,
   Combinatorially Symmetric Cross-Validation (CSCV) and PBO, Deflated
   Sharpe Ratio (DSR), permutation tests, parameter stability surfaces, and
   the gate-composition logic. Plain English at the top of each section,
   formal math below it, references at the bottom.
2. **`demos/cscv_pbo_demo.py`**, the headline demo. Reproduces Bailey,
   Borwein, Lopez de Prado, and Zhu (2014) Algorithm 2.3 on a synthetic
   strategy population. Side-by-side comparison of a disciplined search
   (PBO around 0.15) against pure-noise strategies (PBO around 0.5). The
   pure-noise PBO of 0.5 is the textbook null result and the cleanest
   sanity check on the implementation.
3. **`docs/decision-log-philosophy.md`**, how rejection is institutionalized.
   The branch tournament tracker, forbidden regions, and the per-iteration
   decision-log template that prevents the project from cycling through the
   same falsified hypotheses every quarter.

## Quant signals to look for

- **PBO is computed correctly.** The implementation in the demo enumerates
  C(8, 4) = 70 symmetric partitions, picks the IS-best per partition,
  computes `omega_bar` and `logit`, and returns the fraction of partitions
  with `logit <= 0`. Matches the SSRN paper. Note that PBO is a property of
  the *selection process*, not of any individual strategy; the methodology
  doc spells that out.
- **DSR uses the False Strategy Theorem form.** Expected maximum Sharpe
  under the null is computed with the Euler-Mascheroni form
  `(1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e))`. Skewness
  and kurtosis corrections are in the standard error.
- **Walk-forward is calendar-day, not bar-count.** Sessions and holidays
  are handled correctly. Per-fold WFE is reported, not just the headline
  number.
- **Gate composition is non-negotiable.** A candidate must clear all of
  walk-forward, CSCV, DSR, permutation, parameter stability, and parity
  audits to be considered deployment-eligible. Selectively waiving a gate
  ("the PBO is high but the Sharpe is so good") is explicitly disallowed.
- **The rejected-to-deployed ratio is tracked.** A research process whose
  output is mostly green-light strategies is a warning sign, not a
  strength. The decision log keeps the ratio visible.

## What the public repository does not include

- Live strategy code, real parameters, real PnL numbers, real instrument
  calibrations, or trade-by-trade logs. Those are in a separate private
  repository.
- The full alpha pipeline. The public showcase exhibits the validation
  machinery on synthetic data; the actual research outputs are private.
- Specific instrument coverage and live performance metrics. Available on
  request, under NDA.

## Twenty-minute self-guided walk

1. `README.md` for the framing.
2. `docs/methodology.md` from start to finish. This is the substantive document.
3. `demos/cscv_pbo_demo.py`. Read the code, then run it.
4. `demos/parameter_stability_demo.py` (Tier 2 build). Read it, run it.
5. `docs/parity-engineering.md` for the cross-framework reconciliation
   discipline.
6. `docs/decision-log-philosophy.md` for the governance layer.

If you want to see the private codebase under NDA, contact via email.
