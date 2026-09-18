# Everstock Returns & Refund Decisioning Agent

A personal capstone build — not part of the official Partner Basecamp curriculum in `day1/`/`day2/` — demonstrating the full skill set from the 2-day Claude API course in one cohesive project.

## The business problem

**Everstock**, a mid-size e-commerce retailer, processes thousands of return requests per day. Fully manual review doesn't scale — it's slow and expensive. Fully automatic, rules-based approval is worse: it pays out on fraud and policy violations it never catches (a customer disputing a final-sale classification, a serial-returner pattern, a wrong restocking-fee calculation). The business needs a decision-maker that's grounded in the *actual* order, the *actual* category policy, and the *actual* customer history — every time — and that knows when to say "I can't safely decide this one, a human should."

## What the agent does

Given a return request, it decides **APPROVE / PARTIAL_REFUND / DENY / ESCALATE_TO_HUMAN** by:
1. Looking up the order (`get_order`) — never trusting the customer's description of price, category, or dates.
2. Looking up that category's actual policy (`get_return_policy`) — every time, even for a category it just handled, because the policy text (not its own memory) is the only valid source for the return window, restocking fee, and final-sale rules.
3. Looking up the customer's return history (`get_customer_return_history`) for high-value, electronics, or borderline cases — because fraud signal lives in behavior over time, not in a single order.

It then returns a structured decision with a refund amount, a fraud-risk flag, and a citation that must be traceable to what the tools actually returned — not to a plausible-sounding policy it invented.

## Why this proves the whole course

| Skill | Where it shows up |
|---|---|
| Agentic tool-use loop | The 3-tool loop over order/policy/history |
| Structured output | The final decision, schema-constrained |
| Prompt caching | The policy playbook, cached across every call |
| Model routing | Cheap model for routine returns, stronger model for flagged/high-value ones |
| Effort dial | Applied only on escalation-risk cases |
| Streaming | The final decision call, for a dashboard-style UX |
| Eval harness built from scratch | 8 gold-labeled scenarios + a 2-case holdout, deterministic + LLM-judge graders |
| Accuracy-gated optimization | A "fast but wrong" config can never outscore a correct one |
| Business translation | A before/after table: $/decision, monthly COGS, vs. a manual-review baseline |

The single highest-value test case is `hallucination_trap`: a customer asks about a loyalty-tier grace period that does not exist anywhere in Everstock's real policy. An ungrounded agent will invent one to be helpful. The eval's LLM-judge grader exists specifically to catch that.

> 💸 Running the notebook end to end costs real API usage — the naive `v0` baseline (uncached Opus, 8 scenarios, no schema) is the expensive part by design. Rough order of magnitude: **$1–2.5**.

---

## Files

| File | What it is |
|---|---|
| `Returns_Decisioning_Agent.ipynb` | The notebook — open and run this |
| `Returns_Decisioning_Agent.py` | Canonical source in jupytext percent format. Edit this, then regenerate the notebook with `jupytext --to notebook Returns_Decisioning_Agent.py` so the two never drift. |

## How to run

Uses the same shared root `.env` as every other exercise in this repo (`ANTHROPIC_API_KEY=sk-ant-...`, or Bedrock credentials).

### VS Code / Cursor
1. Open this folder, then `Returns_Decisioning_Agent.ipynb`.
2. Pick the repo's `.venv` Python 3 kernel.
3. **Run All.** Nothing pauses for input — it runs straight through, ending with the business-case table.

### Terminal / Claude Code
```bash
cd returns-decisioning-agent
python3 Returns_Decisioning_Agent.py
```
