# If you are hiring for AI / ML

A two-minute tour of the parts of this repository that demonstrate
responsible use of agentic engineering tools.

## What to look at first

1. **`docs/agentic-research-process.md`**, how AI assistants are
   constrained by deterministic validation gates, written rules, and a
   decision log so that they cannot generate fake research.
2. **`docs/methodology.md`**, the validation gates the AI is constrained
   by. Multiple-testing correction, selection-bias adjustment, parameter
   stability, parity audits. The gates exist regardless of whether AI is in
   the loop, but they are what make AI tooling safe to use.
3. **`docs/decision-log-philosophy.md`**, the persistent memory layer.
   AI assistants run in independent sessions, but the project's memory
   lives in a Markdown log that survives any single session. This is what
   stops an assistant from re-proposing a hypothesis the project rejected
   last month.

## AI / ML signals to look for

- **AI as a research collaborator, not a code generator.** The framing
  matters. The platform uses AI tools to accelerate hypothesis generation,
  literature search, and code review. It does not use them to write
  strategies that bypass validation, and the validation gates have hard
  failure modes that no AI output can paper over.
- **Awareness of the multiple-testing problem.** The Deflated Sharpe Ratio
  exists precisely because AI-assisted research can run thousands of trials
  cheaply, and any naive Sharpe ratio at the end is selection-biased
  garbage. The methodology doc walks through the formula and the
  operational threshold.
- **Anti-hallucination safeguards.** A written constitution forbids silent
  substitutions: if an AI assistant cannot find a requested data source,
  it must stop and ask, not silently use a different one. If it cannot
  prove a hypothesis, it must record the failure in the decision log and
  mark the parameter region as forbidden, not retry under a fresh name.
- **Parity audits as a check on AI-generated code.** Cross-framework
  reconciliation catches not just human bugs but AI-generated bugs. A
  strategy that works in one Python framework but fails in two others is
  rejected, regardless of who or what wrote it.

## What this repository does not include

- A trained ML model. The platform is a research and validation engine, not
  an ML pipeline.
- LLM fine-tuning code, embedding stores, or RAG pipelines. Other projects
  in `github.com/Linja1337` cover transformer training (a 20M-parameter
  language model from scratch in PyTorch) and BERT fine-tuning for
  sentiment analysis on customer reviews.
- "AI-powered" strategy generation, because that phrase is what the gates
  are designed to reject.

## What this repository does demonstrate

- An engineering discipline that treats AI tooling as a power tool with
  guards on it. The guards (CSCV, DSR, permutation tests, parameter
  stability, parity audits, decision logs) exist independently of the AI;
  the AI just makes the velocity higher and the temptation to cut corners
  larger. The discipline is the differentiator.

If your role values rigor about model validation and a clear-eyed view of
what AI tooling is good and bad at, this is the right project to evaluate.
If your role wants a candidate who has shipped a production LLM application,
that lives in a separate repository.

## Ten-minute self-guided walk

1. `README.md`.
2. `docs/agentic-research-process.md`.
3. Skim `docs/methodology.md` for the validation gates.
4. Skim `docs/decision-log-philosophy.md` for the memory model.
5. Run `python demos/cscv_pbo_demo.py` and look at the output.
