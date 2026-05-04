# Decision-log philosophy

A research project that does not remember its own rejected hypotheses will
re-test them every quarter. This document explains the discipline that the
platform uses to prevent that, and why the discipline is more valuable than
any particular result it produces.

## The problem

Quantitative research is a parameter-rich, hypothesis-rich activity. Over a
year, a working researcher will test hundreds of strategy variants. Most will
fail. Many will fail in interesting ways. Without a written record, three
things go wrong:

1. **Repetition.** Six months later, a "new" hypothesis turns out to be a
   minor parameter shift on a strategy that was rejected in March. The work
   re-runs, the rejection re-occurs, the time was wasted.
2. **Drift.** A strategy that was rejected because PBO was too high gets
   re-considered with a slightly different cost model that makes the PBO
   look better. The original reason for rejection is forgotten.
3. **Survivorship illusion.** The set of "live" strategies grows over time,
   while the set of rejected strategies fades from memory. The naive
   inference is that the research process has high yield. It does not. It
   has high yield *for the strategies that survived* and low yield overall.

A decision log fixes all three by making the rejected strategies as visible
as the surviving ones.

## The discipline

The platform's decision log is a single Markdown file with a strict
per-iteration template:

```
### Iteration N: short title

- Hypothesis:       what was expected and why
- Branch (ToT):     which exploration branch this iteration belongs to
- Change made:      what code was modified, with file paths
- Expected outcome: which metrics should improve and by how much
- Actual result:    metric deltas with quantitative comparison
- Conclusion:       confirmed / falsified / inconclusive, with reasoning
- Self-critique:    was the code correct but the market did not cooperate,
                    or was there a logic error?
- Regime note:      what market regime does this result apply to?
- Commit:           git hash
- Next step:        what to try next, informed by this result
- Forbidden region: if falsified, what parameter space should never be
                    revisited without explicit override?
```

Every iteration uses the template. Iterations are append-only. Edits to
prior iterations are forbidden; corrections go in a new iteration that
references the old one.

## Forbidden regions

The mechanism that turns the log from a journal into an active constraint.

When an iteration falsifies a hypothesis, the falsified region of parameter
space (or strategy architecture, or execution-model assumption) is named
explicitly. Future iterations are required to consult the forbidden-region
list before proposing work. A new hypothesis that overlaps with a forbidden
region cannot be acted on without an explicit written justification of why
this attempt should produce a different result.

In practice this looks like:

- "The HA-flip-as-entry pattern is forbidden — falsified at DSR -12.12 in
  Iteration 47. Future hypotheses that use HA flip as an entry trigger must
  cite a fundamentally different confirmation architecture."
- "Re-optimizing strategy 001 as a whole is forbidden — falsified after
  9,553 parameter combinations in the original sweep failed Monte Carlo.
  Future work on strategy 001 must decompose it into atomic components and
  validate each independently."

The forbidden-region list is short, but it carries the weight of the entire
project's prior failures. Each entry represents work that does not have to
be done again.

## Why a Markdown file

Three reasons.

1. **Diffs.** Git makes the change history of the log itself visible. You
   can see when a forbidden region was added, by whom, and in response to
   what evidence. You can revert.
2. **Tool independence.** A Markdown file survives any change in research
   tooling: backtester, optimizer, ML framework, AI assistant. The log is
   not coupled to the system that produced it.
3. **Readable by humans and machines.** The log is the project's onboarding
   document, its postmortem, and the context window that AI assistants load
   at the start of every session. One artifact, three uses.

## The agentic-research connection

This discipline is what makes AI tooling safe to use in the research loop.
An AI assistant has no persistent memory across sessions; without a
decision log, it will cheerfully re-propose a hypothesis the project
rejected last month. With a decision log loaded into context at session
start, the assistant has access to the project's accumulated judgment.

The agentic-research-process document goes deeper into how the assistant is
constrained by the log and the validation gates.

## What this is not

It is not a system of record for trades or PnL. Those live in the broker
statements and the live performance reports. The decision log is about the
research process, not the live results.

It is not a debugging log. Code-level bugs are tracked in commit messages
and issue trackers. The decision log is for hypotheses and their verdicts.

It is not a literature review. References to academic papers and external
research live in `docs/methodology.md` and a separate research-findings
file. The decision log is internal evidence.

## What it costs

Discipline. Five extra minutes per iteration to write the entry properly.
A real cost, paid in writing time, against an unbounded benefit in not
repeating yourself.

A research process whose output includes a thoughtful decision log is also
a research process whose output is auditable. That is what makes it
defensible to a senior reviewer who wants to know not just what works but
why everything else didn't.
