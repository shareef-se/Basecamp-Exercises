# Ridgeline Manufacturing Warranty Root-Cause Agent

The second personal capstone in this repo (first: `../returns-decisioning-agent/`). That build proved tool use, structured output, caching, routing, effort, streaming, an eval harness with an LLM-judge grader, and accuracy-gated optimization — everything from the 2-day course except **context engineering**, since its per-request context never grew large enough to rot. This build closes that gap in a different business domain (manufacturing, not e-commerce), while re-proving everything else — including inference-optimization instrumentation (TTFT/TTC/OTPS) that the first build had dropped.

## The business problem

**Ridgeline Manufacturing** makes precision hydraulic cylinder actuators. When a warranty claim comes in, an engineer must determine whether it's a **known recurring failure pattern** — buried somewhere in a 10,000+ ticket historical corpus, far too large for any context window — or a genuine one-off, then the root cause, warranty coverage, and the recommended action (`REPAIR_UNDER_WARRANTY` / `REPAIR_BILLABLE` / `REPLACE_UNDER_WARRANTY` / `ESCALATE_TO_ENGINEERING`). Catching a real pattern early prevents a costly batch recall; falsely calling a pattern where none exists wastes an engineering escalation. Both failure directions are gradeable, and both are expensive in opposite ways.

## The two context-engineering techniques this build proves

1. **Retrieval instead of context-stuffing** — `search_historical_tickets` returns only the top-5 relevant tickets from the corpus (lightweight keyword-overlap scoring, no embeddings needed), rather than ever putting the whole corpus in context. Proven with a **quantified degradation curve**, not just an anecdote: the same pattern-detection case is run with an increasing raw slice of the corpus stuffed directly into the prompt (5/15/30/50 tickets, no search tool) and the notebook shows whether the agent still finds the real pattern at each size — versus retrieval, which finds it every time regardless of corpus size, at flat cost.
2. **Context editing** — using the real `context-management-2025-06-27` beta (`clear_tool_uses_20250919`), verified against the live API docs before building: the client stays fully stateless (always resend the full history), and the server clears old tool results before processing once a token trigger is crossed, reporting what it cleared in `response.context_management.applied_edits`. Demonstrated on a real multi-round investigation, comparing input-token growth with and without it.

## Architecture — a core library, a lab notebook, and a UI

```
manufacturing-warranty-agent/
├── agent_core.py                        # importable library: tools, schemas, playbook,
│                                         # the agent loop, CONFIG, v0/v1, the eval harness.
│                                         # Import-safe — nothing billed except one cheap
│                                         # credential-check ping.
├── Manufacturing_Warranty_Agent.py       # canonical notebook source (jupytext), imports
│                                         # from agent_core and adds the paid experiment:
│                                         # v0 baseline, eval, context-engineering demos,
│                                         # other levers, v1 sprint, business case.
├── Manufacturing_Warranty_Agent.ipynb    # generated via `jupytext --to notebook`
├── app.py                                # local Streamlit UI, also imports agent_core
└── README.md
```

The split exists because `Manufacturing_Warranty_Agent.py` runs the whole lab as top-level code (~$2-4.5 of API usage). If the UI imported that file directly, every page load would re-trigger the entire experiment. `agent_core.py` holds only reusable definitions, so both the notebook and the UI build on the same logic without either one re-running the other's work.

## The UI

```bash
streamlit run app.py
```

Opens locally at `localhost:8501`. Pick a claim (a preset eval scenario, or a custom claim + message), pick `v0` (naive) or `v1` (optimized) pipeline, and run it — see the tool-call trace and the structured decision live. Scoped deliberately narrow: **it only ever runs one claim at a time**. The full eval suite and optimization sprint stay notebook-only, so a stray click can't trigger a multi-dollar run.

> 💸 Every run in both the notebook and the UI costs real API usage. Full notebook run, rough order of magnitude: **$2–4.5** (the naive Opus `v0` baseline over 9 scenarios is the expensive part, by design). Each single UI run is one claim — a few cents.

## How to run

Uses the same shared root `.env` as every other exercise in this repo.

### The notebook
Open `Manufacturing_Warranty_Agent.ipynb` in VS Code with the repo's `.venv` kernel and Run All — no pauses, ends with the business-case table.

### The UI
```bash
cd manufacturing-warranty-agent
pip install streamlit   # one-time, local dev dependency — not in the repo's root requirements.txt
streamlit run app.py
```

### Terminal
```bash
python3 Manufacturing_Warranty_Agent.py
```
