"""Ridgeline Warranty Root-Cause Agent — interactive UI.

Local page only (`streamlit run app.py`), scoped deliberately narrow: pick one claim,
run it through the agent once, see the trace and decision. It never re-runs the full
eval suite or optimization sprint — those stay notebook-only so a stray click can't
trigger a multi-dollar run. Every run below still costs a real, billed API call.
"""

import streamlit as st

from agent_core import CLAIMS, TASKS, HOLDOUT_TASKS, ridgeline_v0, ridgeline_v1, CONFIG, calculate_cost

st.set_page_config(page_title="Ridgeline Warranty Agent", page_icon="🔧", layout="wide")
st.title("🔧 Ridgeline Warranty Root-Cause Agent")
st.caption("Run one warranty claim through the agent and watch its tool-call trace and decision.")
st.info("Every run below makes real, billed API calls. This page only ever runs **one claim at "
        "a time** — the full eval suite and optimization sprint stay notebook-only on purpose.")

ALL_TASKS = {t["id"]: t for t in TASKS + HOLDOUT_TASKS}

with st.sidebar:
    st.header("1. Pick a claim")
    mode = st.radio("Source", ["Preset scenario", "Custom claim"])

    if mode == "Preset scenario":
        task_id = st.selectbox("Scenario (from the eval set)", list(ALL_TASKS.keys()))
        task = ALL_TASKS[task_id]
        claim = CLAIMS[task["claim_id"]]
        st.text_area("Query sent to the agent", task["query"], height=110, disabled=True)
    else:
        claim_id = st.selectbox("Base claim record", list(CLAIMS.keys()))
        claim = CLAIMS[claim_id]
        message = st.text_area("Customer/technician message", claim["symptom_description"], height=110)
        task = {"id": "custom", "claim_id": claim_id, "query": f"Claim {claim_id}: {message}"}

    st.divider()
    st.subheader("Claim record")
    st.json({k: v for k, v in claim.items() if not k.startswith("_")})

    st.divider()
    st.header("2. Pick a pipeline")
    pipeline_choice = st.radio(
        "Pipeline",
        ["Optimized v1 (routed + cached + effort)", "Naive v0 (uncached Opus, no search)"],
        help="v0 is deliberately naive — it's handed a fixed slice of tickets instead of a "
             "real search tool, and skips structured output. Useful for comparing against v1.",
    )
    run_clicked = st.button("Run agent", type="primary", use_container_width=True)

_ACTION_STYLE = {
    "REPAIR_UNDER_WARRANTY": ("success", "✅"),
    "REPLACE_UNDER_WARRANTY": ("success", "✅"),
    "REPAIR_BILLABLE": ("warning", "💵"),
    "ESCALATE_TO_ENGINEERING": ("error", "🚨"),
}

if run_clicked:
    with st.spinner(f"Running {task['id']} through {'v1' if pipeline_choice.startswith('Optimized') else 'v0'}..."):
        if pipeline_choice.startswith("Optimized"):
            result = ridgeline_v1(task, CONFIG)
        else:
            result = ridgeline_v0(task)

    decision = result["decision"] or {}
    cost = sum(calculate_cost(m, u) for m, u in result["calls"])
    action = decision.get("recommended_action", "UNKNOWN")
    style, icon = _ACTION_STYLE.get(action, ("info", "❔"))

    getattr(st, style)(f"{icon} **{action}**  ·  pattern_match: **{decision.get('pattern_match', '—')}**  ·  "
                       f"warranty_covered: **{decision.get('warranty_covered', '—')}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Elapsed", f"{result['elapsed']:.1f}s")
    col2.metric("Cost", f"${cost:.4f}")
    col3.metric("API calls", len(result["calls"]))
    col4.metric("Tool calls", len(result["tool_calls"]))

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Tool-call trace")
        if not result["tool_calls"]:
            st.write("No tools were called.")
        for i, c in enumerate(result["tool_calls"], 1):
            with st.expander(f"{i}. `{c['name']}({c['arguments']})`"):
                st.code(str(c.get("result", ""))[:2000], language="json")

    with right:
        st.subheader("Final decision")
        st.json(decision)
        if decision.get("matched_ticket_ids"):
            st.caption("Cited historical tickets — cross-check these against the trace above.")
else:
    st.write("Pick a claim and a pipeline in the sidebar, then click **Run agent**.")
