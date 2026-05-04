# If you are hiring for software engineering

A two-minute tour of the parts of this repository that demonstrate engineering
ability, not finance.

## What to look at first

1. **`docs/architecture.md`** — the system view. Layered breakdown, why-decisions
   table, mermaid diagrams of the data flow and the validation pipeline.
2. **`demos/cscv_pbo_demo.py`** — a self-contained Python module that generates
   synthetic data, builds a performance matrix, runs an algorithm with O(C(N, N/2))
   combinatorial complexity, and produces a publication-quality matplotlib figure.
   Run it: `python demos/cscv_pbo_demo.py`.
3. **`demos/parity_audit_demo.py`** — the demonstration of cross-framework
   reconciliation. Two backtesters, one strategy, the same trade list. The
   bug that this technique catches is one of the cleanest examples of why
   integration testing matters in any system, not just trading.

## Engineering signals to look for

- **Vectorization vs sequential split.** The private engine separates indicator
  math (vectorized NumPy) from order routing and stop-loss tracking (Numba
  JIT'd inner loop). The methodology doc and the demos illustrate the pattern.
  Same pattern that turns up in shader code (vertex transforms vectorize,
  fragment writes sequence) and in a thousand other systems.
- **Strict input contracts.** Bar data is normalized to a single schema at
  ingest. Strategies expose `default_params`, `param_space`, `is_valid()`,
  `build_signals()`. Contracts are how you keep a research codebase from
  drifting into spaghetti.
- **Deterministic builds.** Every demo is seeded. Every figure reproduces
  bit-for-bit on a fresh clone. There are no environment variables, no API
  keys, and no data downloads.
- **Decision logs as a software artifact.** The decision-log philosophy doc
  describes how a Markdown journal carries memory across sessions. This is
  the same pattern as a good engineering ADR (Architecture Decision Record),
  applied to research.

## What this repository does not show

- Web frameworks, mobile, or full-stack delivery. Other projects in
  `github.com/Linja1337` cover those.
- Microservice orchestration, Kubernetes, cloud infra. The platform is a
  research engine; deployment is single-host.
- Database schema design at scale. Bar storage is flat-file plus pandas; the
  scale does not justify a database.

If your role requires those, this is the wrong project to evaluate. If your
role values numerical engineering, perf-conscious Python, and disciplined
research process, this is the right one.

## Five-minute self-guided walk

1. Open `README.md`. Read the top section.
2. Open `docs/architecture.md`. Skim the diagrams.
3. Open `demos/cscv_pbo_demo.py`. The whole file is meant to be readable.
4. Run it: `pip install -r requirements.txt && python demos/cscv_pbo_demo.py`.
5. Open the generated `demos/output/cscv_pbo.png`.

If anything in the code looks unclear or wrong, that is a question for an
interview. The code is the resume here.
