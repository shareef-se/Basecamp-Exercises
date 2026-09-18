# %% [markdown]
# # Ridgeline Manufacturing — Warranty Root-Cause Agent
#
# **Ridgeline Manufacturing** makes precision hydraulic cylinder actuators. When a warranty
# claim comes in, an engineer must determine whether it's a **known recurring failure
# pattern** — buried somewhere in a 10,000+ ticket historical corpus, far too large for any
# context window — or a genuine one-off, then the root cause, warranty coverage, and the
# recommended action. Catching a real pattern early prevents a costly batch recall; falsely
# calling a pattern where none exists wastes an engineering escalation.
#
# This is the **second** capstone in this repo (first: `../returns-decisioning-agent/`). That
# one proved tool use, structured output, caching, routing, effort, streaming, an eval
# harness with an LLM-judge grader, and accuracy-gated optimization — everything except
# **context engineering**, because its per-request context was too small to ever rot. This
# build closes that gap with two real techniques (retrieval instead of context-stuffing, and
# context editing) while re-proving everything else, including inference-optimization
# instrumentation (TTFT/TTC/OTPS) that the first build had dropped.
#
# All the reusable machinery — tools, schemas, the playbook, the agentic loop, `CONFIG`,
# `v0`/`v1`, the eval harness — lives in `agent_core.py` so the same logic backs this
# notebook, the terminal script, and the `app.py` UI without tripling the code (and without
# a UI page accidentally re-running the whole paid experiment on every load).
#
# > 💸 Running every cell costs real API usage. Rough order of magnitude: **$2–4.5** — the
# > naive Opus `v0` baseline over 9 scenarios is the expensive part, by design.

# %% [markdown]
# # Part 0 · Setup
#
# Everything below imports from `agent_core.py`, which handles credentials, the client, the
# tool/data/schema definitions, and prints its own readiness banners as it loads.

# %%
from agent_core import (
    client, MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS,
    TODAY, WARRANTY_TERMS, CLAIMS, HISTORICAL_TICKETS, PATTERN_TICKET_IDS, RED_HERRING_TICKET_IDS,
    V0_STUFFED_SLICE, PLAYBOOK, CACHED_SYSTEM,
    TOOL_SPECS, TOOL_SPECS_V0, DECISION_SCHEMA,
    run_tool_loop, collect_tool_calls, final_decision_call, decide_efficient, run_tool_loop_ce,
    triage, ridgeline_v0, ridgeline_v1, CONFIG,
    PRICING, calculate_cost,
    GRADER_REGISTRY, TASKS, HOLDOUT_TASKS, run_batch, print_report,
    text_of, extract_json,
)
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional
from tabulate import tabulate

print("Notebook imports ready")

# %% [markdown]
# # Part 1 · The instrument panel
#
# "Measure first" from the inference-optimization lab, restored properly: **TTFT** (time to
# first token), **OTPS** (output tokens per second, measured over generation time only), and
# **TTC** (total completion time) are three different numbers with three different fixes —
# collapsing them into one `elapsed` figure (as the first capstone did) hides that. This is
# the same `BenchmarkResult`/`_stream_request`/`compute_otps` shape as `Inference_Optimization.py`.

# %%
@dataclass
class BenchmarkResult:
    ttft: float
    total_time: float
    input_tokens: int
    output_tokens: int
    model: str
    test_name: str
    otps: Optional[float] = None
    cost: Optional[float] = None


@dataclass
class BenchmarkSuite:
    results: List[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def summary(self) -> str:
        groups = {}
        for r in self.results:
            groups.setdefault(r.test_name, []).append(r)
        rows = []
        for name, group in groups.items():
            rows.append([name, len(group),
                        f"{statistics.mean(r.ttft for r in group) * 1000:.0f}",
                        f"{statistics.mean(r.total_time for r in group) * 1000:.0f}",
                        f"{statistics.mean(r.otps or 0 for r in group):.1f}",
                        f"${sum(r.cost or 0 for r in group) * 1000:.2f}"])
        return tabulate(rows, headers=["Test", "Runs", "TTFT(ms)", "TTC(ms)", "OTPS", "$/1K calls"], tablefmt="grid")


def _stream_request(messages, model, max_tokens=256, system=None):
    ttft = None
    params = dict(model=model, max_tokens=max_tokens, messages=messages)
    if system is not None:
        params["system"] = system
    start = time.perf_counter()
    with client.messages.stream(**params) as stream:
        for event in stream:
            if ttft is None and event.type == "content_block_start":
                ttft = time.perf_counter() - start
        response = stream.get_final_message()
    total = time.perf_counter() - start
    return ttft if ttft is not None else total, total, response


def compute_otps(ttft, total_time, output_tokens):
    gen_time = max(total_time - ttft, 1e-9)
    return output_tokens / gen_time


suite = BenchmarkSuite()
PROBE = "In one sentence, what causes hydraulic seal extrusion?"
for model_id, label in [(MODEL_HAIKU, "haiku"), (MODEL_SONNET, "sonnet"), (MODEL_OPUS, "opus")]:
    for _ in range(2):
        ttft, total, resp = _stream_request([{"role": "user", "content": PROBE}], model=model_id)
        otps = compute_otps(ttft, total, resp.usage.output_tokens)
        suite.add(BenchmarkResult(ttft=ttft, total_time=total, input_tokens=resp.usage.input_tokens,
                                  output_tokens=resp.usage.output_tokens, model=model_id, test_name=label,
                                  otps=otps, cost=calculate_cost(model_id, resp.usage)))

print(suite.summary())
print("\nThis is what 'measure first' means before touching any lever — TTFT, TTC, and OTPS "
      "are different failure modes with different fixes.")

# %% [markdown]
# # Part 2 · Try the agent live
#
# Two non-eval demo claims (never used for grading, so this trace can't leak an answer key):
# a straightforward one-off, and a long-service-life wear-out.

# %%
def print_trace(result: dict, label: str):
    print(f"\n── {label} " + "─" * max(1, 60 - len(label)))
    for c in result["tool_calls"]:
        print(f"  [tool_use] {c['name']}({c['arguments']}) -> {str(c.get('result', ''))[:160]}")
    print(f"  [decision] {json.dumps(result['decision'], indent=2)}")
    print(f"  {len(result['calls'])} API calls · {result['elapsed']:.1f}s")


demo1 = decide_efficient("Claim CLM-9001: cylinder won't fully retract, seems to bind partway.",
                         model=MODEL_SONNET, system=PLAYBOOK)
print_trace(demo1, "CLM-9001 — one-off binding issue")

demo2 = decide_efficient("Claim CLM-9002: some seal seepage, unit's been in service a long time.",
                         model=MODEL_SONNET, system=PLAYBOOK)
print_trace(demo2, "CLM-9002 — long-service-life wear-out")

# %% [markdown]
# # Part 3 · Naive v0 — the inherited baseline
#
# `ridgeline_v0` (in `agent_core.py`): one expensive uncached model, no search tool — a fixed
# 14-ticket slice of the corpus is stuffed into the prompt instead (an engineer handed a
# printout instead of a real search tool), then an unconstrained essay-then-JSON pass. We
# already confirmed that slice contains **zero of the 5 real pattern tickets and both red
# herrings** — so this baseline is built to fail the needle-in-haystack case and risk the
# false-positive trap, not by luck.

# %%
print("Running Ridgeline v0 on the 9 in-sample scenarios (sequential, Opus, uncached, "
      "stuffed-context instead of search)...")
BASELINE = run_batch(ridgeline_v0, TASKS)
print_report(BASELINE, "Ridgeline v0 — the naive baseline")

# %% [markdown]
# ### Is this eval trustworthy?
#
# Check the printed failures above. `needle_in_haystack_pattern_found` should fail — v0 never
# saw the real pattern tickets. `false_positive_trap` is a real risk, not a guarantee — v0's
# slice contains both red herrings, so it may confidently claim a pattern that a proper
# investigation would rule out. If v0 passed everything, the dataset isn't hard enough yet.
#
# Manual judge-sanity check worth running once by hand: take a transcript where
# `no_false_pattern_claim` passed, hand-inject a fabricated ticket ID into
# `matched_ticket_ids`, re-run just that grader, and confirm it flips to FAIL.

# %% [markdown]
# # Part 4 · Eval harness recap
#
# The harness itself (`GRADER_REGISTRY`, `run_batch`, `print_report`) lives in `agent_core.py`
# — same synthesis as the first capstone: one runner produces both pass/fail grading *and*
# cost/latency aggregation, so it serves the eval above *and* the accuracy-gated scorecard in
# Part 8. Five graders: `decision_match`, `pattern_match`, `matched_tickets_overlap`,
# `tool_called` (all deterministic), and `no_false_pattern_claim` (LLM-judge, purpose-built
# for the false-positive trap — the direct analog of the returns agent's hallucination grader).

# %% [markdown]
# # Part 5 · Context-engineering levers
#
# The differentiator vs. the first capstone. Two distinct techniques for two distinct
# problems — demonstrated separately so the difference is visible, not just described.

# %% [markdown]
# ## 5a — Retrieval vs. context-stuffing: a quantified degradation curve
#
# Instead of a single before/after anecdote, feed the agent an increasing *raw, unfiltered*
# slice of the corpus directly in context (no search tool) on the same pattern-detection
# case, and check whether it still correctly identifies the Lot 4471 pattern at each size.
# This is the actual "context rot" methodology — a curve as input grows, not one data point.

# %%
def stuffed_context_decide(claim_query: str, n_tickets: int, model=MODEL_SONNET) -> dict:
    """No search tool at all — just an increasing raw slice of the ticket corpus in the
    system prompt, exactly like ridgeline_v0 but with a tunable slice size."""
    tickets = list(HISTORICAL_TICKETS.values())[:n_tickets]
    slice_text = "\n\n".join(
        f"[{t['ticket_id']} / {t['part_number']}] {t['technician_free_text']} "
        f"(root cause on file: {t['root_cause_code']})" for t in tickets)
    system = PLAYBOOK + "\n\n## Reference: ticket excerpts (not exhaustive, not searchable)\n" + slice_text
    t0 = time.perf_counter()
    messages, calls = run_tool_loop(claim_query, model, system, tools=TOOL_SPECS_V0)
    decision, final_resp = final_decision_call(messages, model, system, tools=TOOL_SPECS_V0)
    calls.append((model, final_resp.usage))
    input_tokens = sum(u.input_tokens for _, u in calls)
    return {"decision": decision, "n_tickets": n_tickets, "input_tokens": input_tokens,
            "cost": sum(calculate_cost(m, u) for m, u in calls), "elapsed": time.perf_counter() - t0}


needle_query = TASKS[1]["query"]  # needle_in_haystack_pattern_found
rows = []
for n in (5, 15, 30, 50):
    r = stuffed_context_decide(needle_query, n_tickets=n)
    found = r["decision"] and r["decision"].get("pattern_match") in ("SUSPECTED_EMERGING", "CONFIRMED_RECURRING") \
            and bool(set(r["decision"].get("matched_ticket_ids") or []) & PATTERN_TICKET_IDS)
    rows.append([n, r["input_tokens"], r["decision"] and r["decision"].get("pattern_match"),
                "YES" if found else "no", f"${r['cost']:.4f}"])
print(tabulate(rows, headers=["Tickets stuffed", "Input tokens", "pattern_match", "Correctly found?", "Cost"], tablefmt="simple"))
print("\nCompare this to search-based retrieval, which finds the pattern regardless of corpus "
      "size because it only ever returns the 5 most relevant tickets, not more tokens as the "
      "corpus grows. That's the actual finding 'context rot' research describes: relevance, "
      "not just length, is the problem — feeding more raw text doesn't help once the useful "
      "signal is diluted, and can actively make it worse by burying it further.")

with_search = decide_efficient(needle_query, model=MODEL_SONNET, system=PLAYBOOK)
found_with_search = bool(set((with_search["decision"] or {}).get("matched_ticket_ids", [])) & PATTERN_TICKET_IDS)
print(f"\nWith real search (agent_core's default tools): pattern found = {found_with_search}, "
      f"input across all calls = {sum(u.input_tokens for _, u in with_search['calls'])} tokens, "
      f"cost = ${sum(calculate_cost(m, u) for m, u in with_search['calls']):.4f} "
      f"— flat regardless of how large the underlying corpus actually is.")

# %% [markdown]
# ## 5b — Context editing: clearing old tool results mid-investigation
#
# Verified against the live API docs, not assumed: the client stays **fully stateless** —
# send the complete, unedited `messages` history every turn. The server clears old
# tool_use/tool_result pairs *before* processing once a token trigger is crossed, and reports
# what it cleared in `response.context_management.applied_edits`. Nothing needs to be echoed
# back or resynced (unlike compaction). The default trigger is 100,000 input tokens — far
# above what this small demo would ever reach — so the demo below sets a low trigger
# explicitly, or the lever would silently never fire.

# %%
ce_query = TASKS[1]["query"]  # the same multi-round investigation case as 5a

print("Baseline (no context editing) — full history resent every turn:")
_, base_calls = run_tool_loop(ce_query, MODEL_SONNET, PLAYBOOK)
for i, (m, u) in enumerate(base_calls, 1):
    print(f"  turn {i}: input_tokens={u.input_tokens}")

print(f"\nWith context editing (trigger={CONFIG['context_edit_trigger_tokens']} input tokens, keep=2 tool uses):")
_, ce_calls, edit_log = run_tool_loop_ce(ce_query, MODEL_SONNET, PLAYBOOK,
                                        trigger_tokens=CONFIG["context_edit_trigger_tokens"], keep_tool_uses=2)
for i, (m, u) in enumerate(ce_calls, 1):
    print(f"  turn {i}: input_tokens={u.input_tokens}")
if edit_log:
    for e in edit_log:
        print(f"  -> cleared on turn {e['turn']}: {e['applied_edits']}")
else:
    print("  (no edit fired — this investigation didn't cross the trigger; still shows the mechanism is off by default)")

print("\nRetrieval (5a) is a relevance filter applied at read time. Context editing (5b) is a "
      "hygiene pass over facts already read. Related tools for a related problem, not the "
      "same lever — a long investigation needs both.")

# %% [markdown]
# # Part 6 · Other optimization levers

# %% [markdown]
# ## Lever 1 — Prompt caching

# %%
for label in ("COLD (writes cache)", "WARM (reads cache)"):
    resp = client.messages.create(model=MODEL_SONNET, max_tokens=200, system=CACHED_SYSTEM,
                                  tools=TOOL_SPECS, messages=[{"role": "user", "content": TASKS[0]["query"]}])
    u = resp.usage
    print(f"{label:20s} uncached_in={u.input_tokens:>5} · cache_write={u.cache_creation_input_tokens or 0:>5} · "
          f"cache_read={u.cache_read_input_tokens or 0:>5} · cost=${calculate_cost(MODEL_SONNET, u):.5f}")

# %% [markdown]
# ## Lever 2 — Model routing

# %%
rows = []
for t in TASKS:
    claim = CLAIMS[t["claim_id"]]
    verdict, resp = triage(claim, t["query"])
    rows.append([t["id"], verdict, f"${calculate_cost(MODEL_HAIKU, resp.usage):.5f}"])
print(tabulate(rows, headers=["Task", "Triage", "Triage cost"], tablefmt="simple"))

# %% [markdown]
# ## Lever 3 — Output discipline + effort, Lever 4 — Streaming

# %%
efficient_demo = decide_efficient(TASKS[0]["query"], model=MODEL_SONNET, system=CACHED_SYSTEM, effort="low")
v0_row = BASELINE["rows"][0]
print(f"Efficient single-pass: {efficient_demo['elapsed']:.1f}s · {len(efficient_demo['calls'])} calls · "
      f"${sum(calculate_cost(m, u) for m, u in efficient_demo['calls']):.5f}")
print(f"v0 on the same task:   {v0_row['elapsed']:.1f}s · {v0_row['calls']} calls · ${v0_row['cost']:.4f}")

_stream_messages, _ = run_tool_loop(TASKS[3]["query"], MODEL_SONNET, CACHED_SYSTEM)
# The loop always ends on an assistant turn; append one user turn before the next call,
# same fix as final_decision_call in agent_core.py — see its docstring for why.
_stream_messages = _stream_messages + [{"role": "user", "content":
    "Based on everything you've found, provide your final decision now, matching the required schema."}]
_t0 = time.perf_counter()
_ttft = None
with client.messages.stream(model=MODEL_SONNET, max_tokens=700, system=CACHED_SYSTEM, tools=TOOL_SPECS,
                            tool_choice={"type": "none"}, messages=_stream_messages,
                            output_config={"format": DECISION_SCHEMA}) as stream:
    for event in stream:
        if _ttft is None and event.type == "content_block_start":
            _ttft = time.perf_counter() - _t0
    _ = stream.get_final_message()
_total = time.perf_counter() - _t0
print(f"\nStreamed final decision call:  TTFT {_ttft*1000:.0f}ms · TTC {_total*1000:.0f}ms")

# %% [markdown]
# # Part 7 · Optimized v1, holdout, and the business case
#
# `CONFIG` (in `agent_core.py`) is the one dial left un-hardcoded — everything else here is a
# complete, working reference rather than a stubbed build-along.

# %%
def engagement_score(report: dict, baseline: dict) -> int:
    if report["accuracy"] < baseline["accuracy"]:
        return 0
    return round(50 * baseline["cost_per_decision"] / max(report["cost_per_decision"], 1e-9)
                 + 50 * baseline["p50_s"] / max(report["p50_s"], 1e-9))


def run_scorecard(config, label, tasks=TASKS):
    report = run_batch(lambda t: ridgeline_v1(t, config), tasks, workers=config.get("parallel_workers", 0))
    print_report(report, label)
    score = engagement_score(report, BASELINE)
    gate = "" if report["accuracy"] >= BASELINE["accuracy"] else "  ⛔ accuracy gate failed — score zeroed"
    print(f"\n  ENGAGEMENT SCORE: {score}  (v0 baseline = 100){gate}")
    return report


print("Sequential v1:")
run_scorecard({**CONFIG, "parallel_workers": 0}, "Ridgeline v1 — sequential")

print("\nParallel v1 (same CONFIG, 4 workers):")
OPTIMIZED = run_scorecard(CONFIG, "Ridgeline v1 — parallel x4")

# %% [markdown]
# ### Holdout — did this generalize?

# %%
RUN_HOLDOUT = True
if RUN_HOLDOUT:
    run_scorecard(CONFIG, "Ridgeline v1 — HOLDOUT", tasks=HOLDOUT_TASKS)
else:
    print("Holdout armed. Set RUN_HOLDOUT = True once CONFIG is final.")

# %% [markdown]
# ### The business case

# %%
ASSUMPTIONS = {
    "daily_claim_volume": 400,
    "engineer_review_minutes_per_claim": 12,
    "engineer_hourly_usd": 65,
    "cost_per_missed_pattern_usd": 250000,  # a pattern that ships another 500 units before it's caught
}


def business_case(baseline: dict, optimized: dict, a=ASSUMPTIONS):
    monthly_volume = a["daily_claim_volume"] * 30
    ai_cogs_v0 = baseline["cost_per_decision"] * monthly_volume
    ai_cogs_v1 = optimized["cost_per_decision"] * monthly_volume
    manual_cogs = (a["engineer_hourly_usd"] / 60 * a["engineer_review_minutes_per_claim"]) * monthly_volume

    rows = [
        ["Accuracy (audited checks)", f"{baseline['accuracy']*100:.0f}%", f"{optimized['accuracy']*100:.0f}%", "gate held"],
        ["p50 latency / decision", f"{baseline['p50_s']:.1f}s", f"{optimized['p50_s']:.1f}s",
         f"{baseline['p50_s']/max(optimized['p50_s'],1e-9):.1f}x faster"],
        ["Cost / decision", f"${baseline['cost_per_decision']:.4f}", f"${optimized['cost_per_decision']:.4f}",
         f"{baseline['cost_per_decision']/max(optimized['cost_per_decision'],1e-9):.1f}x cheaper"],
        ["Monthly AI COGS (est. volume)", f"${ai_cogs_v0:,.0f}", f"${ai_cogs_v1:,.0f}", f"${ai_cogs_v0-ai_cogs_v1:,.0f} saved"],
        ["vs. fully-manual engineer review", f"${manual_cogs:,.0f}/mo", f"${manual_cogs:,.0f}/mo",
         f"v1 is {manual_cogs/max(ai_cogs_v1,1e-9):.0f}x cheaper than manual"],
        ["Cost of one missed recurring pattern", f"${a['cost_per_missed_pattern_usd']:,.0f}", "—",
         "the reason pattern_match recall, not just cost, is the KPI that matters here"],
    ]
    print("RIDGELINE WARRANTY ROOT-CAUSE — before / after")
    print(tabulate(rows, headers=["Metric", "v0 (naive)", "Optimized v1", "Delta"], tablefmt="grid"))
    print("\nAssumptions: " + ", ".join(f"{k}={v}" for k, v in a.items()))


business_case(BASELINE, OPTIMIZED)

# %% [markdown]
# ---
# ## What this demonstrates, end to end
#
# | Skill | Where |
# |---|---|
# | Agentic tool-use loop | `agent_core.run_tool_loop` |
# | Structured output | `DECISION_SCHEMA`, `final_decision_call` |
# | Inference instrumentation (TTFT/TTC/OTPS) | Part 1 |
# | **Retrieval vs. context-stuffing (quantified)** | Part 5a |
# | **Context editing** | Part 5b |
# | Prompt caching | Part 6 Lever 1 |
# | Model routing | Part 6 Lever 2 |
# | Effort dial + streaming | Part 6 Levers 3-4 |
# | Eval harness from scratch | `agent_core.GRADER_REGISTRY`/`run_batch`, Part 3-4 |
# | LLM-as-judge grading | `grade_no_false_pattern_claim` |
# | Accuracy-gated optimization | Part 7 `engagement_score` |
# | Business translation | Part 7 `business_case` |
# | **A UI to run it interactively** | `app.py` |
