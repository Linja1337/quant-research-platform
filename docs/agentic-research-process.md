# Agentic research process

How AI tooling is used inside the platform without polluting the research
output. The short version: AI is a velocity tool, the validation gates are
the truth tools, and the discipline is what stops the first one from
overrunning the second.

## The premise

Modern AI assistants can write code, search literature, summarize papers,
and propose hypotheses faster than a single researcher working alone. That
velocity is real. So is the failure mode it enables: a thousand strategies
generated, a hundred backtested, ten with apparent edge, none with an
actual edge. The Deflated Sharpe Ratio formula makes this concrete: at
1,000 trials, the expected maximum Sharpe under the null is roughly 3.7.
Anything below that is statistically indistinguishable from luck. AI
tooling makes 1,000 trials cheap. The gates have to compensate.

This document is about that compensation.

## What AI is used for

Three things. None of them are "generate a profitable strategy."

1. **Literature search.** Surface papers, blog posts, and reference
   implementations relevant to a hypothesis. Summarize, cite, link.
2. **Code review and refactoring.** Read existing code, propose
   simplifications, catch bugs. Most useful in the parity-audit work,
   where the same logic appears in multiple frameworks and consistency
   matters.
3. **Hypothesis brainstorming.** Generate divergent ideas for what to try
   next, including ideas the human researcher might not have considered.
   The output is treated as a candidate list, not a recommendation.

What AI is *not* used for:

- Writing the validation code. CSCV, DSR, walk-forward, permutation, and
  parameter-stability implementations are human-written, peer-reviewable,
  and tied to specific equations from the literature.
- Picking deployment-eligible strategies. The gates pick those, on metric
  evidence alone.
- Bypassing the decision log. Every AI-generated hypothesis goes through
  the same per-iteration template as a human-generated hypothesis. The
  log entry is the gate that prevents AI velocity from creating a flood
  of un-recorded work.

## The constraints

Three rules, written down, that shape every AI session:

### Rule 1. No silent substitutions.

If an AI assistant cannot fetch a requested data source, it must stop and
ask, not silently substitute a different one. If an instrument list
includes one ticker that is not available on the venue, the assistant
must report which ticker is missing and ask whether to skip it, change
venues, or stop.

The reason: silent substitutions invalidate the research question. A
strategy designed to be tested on five tickers and silently tested on four
of them produces a result that does not answer the original question and
that is hard to detect downstream.

### Rule 2. No hypothesis revival without justification.

Before any new hypothesis is acted on, the assistant consults the
forbidden-region list (see `decision-log-philosophy.md`). If the
hypothesis overlaps a forbidden region, the assistant must explicitly
state why this attempt is expected to produce a different result. The
human reviewer accepts or rejects the justification.

The reason: an assistant with no persistent memory will cheerfully re-
propose a hypothesis the project rejected last month. The forbidden-region
check loads the project's memory back into the session.

### Rule 3. The validation gates are non-negotiable.

A strategy candidate must pass walk-forward, CSCV, DSR, permutation,
parameter stability, and parity audits, regardless of who or what proposed
it. There is no "the AI is confident this works" exemption. The gates have
hard failure modes (PBO > 0.10 fails, DSR < 0.95 fails, parity mismatch
fails) that no AI confidence score can paper over.

The reason: the gates exist precisely to filter out plausible-looking
results that turn out to be selection bias. Any exemption defeats them.

## The session pattern

A typical AI-assisted research session opens with the assistant loading:

- The constitution (a written document of the rules above, plus
  organizational context)
- The current state of the decision log (last 200 lines)
- The current set of forbidden regions
- The relevant strategy file or research artifact

This loading is the first thing the assistant does, before any new work.
The constitution is the rule set, the decision log is the memory, and the
forbidden regions are the negative-space map of what not to retry. With
all three loaded, the assistant operates inside a deterministic envelope.

Without all three loaded, the assistant is just a fast junior researcher
with no memory. That mode produces enthusiastic, plausible-looking,
re-derivative work. The session pattern exists to prevent it.

## The anti-pattern this avoids

The 2026 quant-research failure mode is straightforward to describe:

1. Researcher with AI assistant runs 5,000 strategy variants in a week.
2. The top 10 by Sharpe look impressive.
3. Researcher writes a memo highlighting them.
4. Assistant cleans up the memo and adds confident-sounding language.
5. Memo lands on a senior reviewer's desk.
6. Senior reviewer asks "what was the multiple-testing correction?" and
   "what was the parity audit?" and "where is the decision log entry for
   the 4,990 that were rejected?"
7. There are no answers.
8. The memo is binned, the strategies are not deployed, the time was wasted.

The platform's discipline produces a different artifact. The decision log
shows the 4,990 rejections with reasons. The DSR explains why even the top
10 are statistically indistinguishable from noise at that trial count. The
parity audit confirms (or fails) that the implementation is correct. The
memo, when it exists, is short and points at the artifacts.

## What this discipline costs

It is slower per session. Loading the constitution, the decision log, and
the forbidden regions takes time. Writing the per-iteration log entry
takes time. Running the gates takes time.

It is faster per result. The discipline collapses the gap between "this
looks promising" and "this is deployable" because the work that closes the
gap is built into the process. A candidate that emerges from a disciplined
session is already audit-ready.

The trade is between session-level velocity and project-level velocity.
The platform optimizes for project-level.


