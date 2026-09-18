# %% [markdown]
# # Everstock Returns & Refund Decisioning Agent
#
# **Everstock** is a mid-size e-commerce retailer processing thousands of return requests
# per day. Fully manual review is too slow and too expensive to sustain at that volume.
# Fully automatic, rules-based approval is worse: it pays out on fraud and policy
# violations it never catches. Someone — or something — has to read each request, look up
# the actual order and policy, and make a defensible call.
#
# This notebook builds that agent end to end, using every skill from the 2-day Claude API
# course: an agentic tool-use loop, structured output, prompt caching, model routing, a
# from-scratch eval harness (deterministic + LLM-as-judge graders), and a cost/latency
# optimization pass gated on accuracy — closing with a business case a steering committee
# could actually act on.
#
# **The build in three acts:**
# 1. **Parts 1–3 — The scenario, the tools, the agent.** Everstock's return policy, sample
#    orders and customer history, the three tools, and a naive-but-correct agentic loop.
# 2. **Part 4 — The eval harness.** Deterministic graders plus an LLM-judge grader built
#    specifically to catch the costliest failure mode: inventing a policy clause or an order
#    fact that doesn't exist. Run against a deliberately wasteful `v0` baseline and confirm
#    the eval actually has teeth before trusting it for anything downstream.
# 3. **Parts 5–6 — Optimize, then prove it.** Prompt caching, model routing, structured
#    single-pass output with an effort dial, streaming, and parallelism — each measured
#    before/after — then an optimized `v1`, a 2-case holdout check, and a before/after
#    business-case table.
#
# > 💸 Running every cell costs real API usage — a naive Opus baseline over 8 scenarios is
# > the most expensive part, by design. Rough order of magnitude: **$1–2.5**.

# %% [markdown]
# # Part 0 · Setup
#
# Same setup pattern as the rest of the course: dependencies, then your API key. Never
# paste a key into a cell.

# %%
# Install dependencies into THIS kernel — safe to re-run; survives locked-down (PEP 668) Pythons.
import importlib.util, subprocess, sys

def _ensure_packages(requirements):
    """requirements: list of (import_name, pip_spec). Install only what is missing,
    into the running interpreter. Tries a normal install, then user-space, then a
    PEP 668 override (user-space first, system-wide only as a last resort)."""
    missing = [pip for mod, pip in requirements if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print("Installing " + ", ".join(missing) + " — first run only, please wait…", flush=True)
    base = [sys.executable, "-m", "pip", "install", "-q"]
    last = None
    for extra in ([], ["--user"], ["--user", "--break-system-packages"], ["--break-system-packages"]):
        last = subprocess.run(base + extra + missing, capture_output=True, text=True)
        if last.returncode == 0:
            return
    pip_said = (last.stderr or last.stdout or "").strip().splitlines() if last else []
    tail = "\n      ".join(pip_said[-3:]) if pip_said else "(no output from pip)"
    raise SystemExit(
        "\n  Couldn't install: " + ", ".join(missing) + "\n"
        "  This Python is locked down (PEP 668) or offline. Quickest fix is a venv:\n"
        f"      {sys.executable} -m venv .venv\n"
        "      source .venv/bin/activate          # Windows: see SETUP.md\n"
        f"      pip install {' '.join(missing)}\n"
        "  Then pick the .venv interpreter in VS Code (kernel picker, top-right) and Run All.\n"
        f"  (pip said: {tail})\n"
    )

_ensure_packages([("anthropic", "anthropic"), ("tabulate", "tabulate")])
print("✓ Dependencies ready")

import os
import json
import time
import statistics
from datetime import date
from dataclasses import dataclass, field
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from tabulate import tabulate

import anthropic
from anthropic.types import ToolUseBlock, TextBlock

# %% [markdown]
# ### Setup — connect to Claude
#
# Creates a gitignored **`.env`** file the first time you run it. Paste your key after
# `ANTHROPIC_API_KEY=`, save, and re-run — a green **"✓ API key verified"** banner confirms
# you're connected.

# %%
def _status(ok, msg):
    """Green/red banner in notebooks; plain text when run as a script."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
            raise RuntimeError("not in a notebook kernel - use the plain-text banner")
        from IPython.display import display, HTML
        color = "#1a7f37" if ok else "#b42318"
        bg = "#e6f4ea" if ok else "#fdecea"
        icon = "✓" if ok else "✗"
        display(HTML(
            f'<div style="padding:12px 16px;border-radius:8px;background:{bg};'
            f'border:1.5px solid {color};color:{color};font-weight:600;'
            f'font-size:15px;font-family:sans-serif;">{icon} {msg}</div>'
        ))
    except Exception:
        print(("[OK] " if ok else "[!!] ") + msg)

import pathlib

_ENV_TEMPLATE = (
    "# Anthropic API key — paste after the = (no quotes, no spaces), then save and\n"
    "# re-run the setup cell. Get one at https://console.anthropic.com/\n"
    "ANTHROPIC_API_KEY=paste-your-key-here\n"
    "\n"
    "# --- Using Amazon Bedrock instead? Comment out the line above and fill these in:\n"
    "# AWS_BEARER_TOKEN_BEDROCK=paste-your-bedrock-api-key-here\n"
    "# AWS_REGION=us-east-1\n"
)


def _resolve_env_file():
    """Nearest existing .env walking up from the working dir (one root .env serves every
    exercise in this repo); if none exists yet, point at the repo root."""
    here = pathlib.Path.cwd().resolve()
    for d in [here, *here.parents]:
        if (d / ".env").is_file():
            return d / ".env"
    root = next((d for d in [here, *here.parents]
                 if (d / "SETUP.md").exists() or (d / ".git").exists()), here)
    return root / ".env"


_env_file = _resolve_env_file()
if not _env_file.exists():
    _env_file.write_text(_ENV_TEMPLATE)
    print(f"Created {_env_file.name} in {_env_file.parent} — open it, add your key, "
          "save, then re-run this cell.")

_file = {}
for _line in (_env_file.read_text().splitlines() if _env_file.exists() else []):
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        _file[_k.strip()] = _v.strip().strip('"').strip("'")
for _k, _v in _file.items():
    if _k != "ANTHROPIC_API_KEY":
        os.environ.setdefault(_k, _v)

_shell_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
_anthropic_key = _shell_key if _shell_key.startswith("sk-ant-") else _file.get("ANTHROPIC_API_KEY", "").strip()
_bedrock_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
_bedrock_region = os.environ.get("AWS_REGION", "").strip()

if _anthropic_key.startswith("sk-ant-"):
    PROVIDER = "anthropic"
elif _bedrock_token:
    PROVIDER = "bedrock"
else:
    PROVIDER = None


def _needs_credentials(head, body):
    _shown = False
    try:
        from IPython import get_ipython
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            import html as _html
            from IPython.display import HTML, display
            display(HTML(
                '<div style="padding:12px 16px;border-radius:8px;background:#fff8c5;'
                'border:1.5px solid #9a6700;font-size:15px;font-family:sans-serif;">'
                '<div style="color:#9a6700;font-weight:600;">' + _html.escape(head) + '</div>'
                '<pre style="margin:10px 0 0;font-family:inherit;font-size:14px;font-weight:400;'
                'color:#141413;white-space:pre-wrap;">' + _html.escape(body) + '</pre></div>'
            ))
            _shown = True
    except Exception:
        pass
    if not _shown:
        print("\n" + head + ":\n   " + body.replace("\n", "\n   ") + "\n")
    raise SystemExit("Credentials missing — see the message above.")


if PROVIDER is None:
    _needs_credentials(
        "📋 Add your credentials to continue",
        f"Open this file:  {_env_file}\n\nUsing the Anthropic API? Set:\n"
        "    ANTHROPIC_API_KEY=sk-ant-...\n\nUsing Amazon Bedrock? Set both:\n"
        "    AWS_BEARER_TOKEN_BEDROCK=<your Bedrock API key>\n"
        "    AWS_REGION=us-east-1          # the region your models are enabled in\n"
        "\nSave the file, then click ▶ on this cell again."
    )

if PROVIDER == "bedrock" and not _bedrock_region:
    _needs_credentials(
        "📋 Bedrock needs a region",
        f"Found AWS_BEARER_TOKEN_BEDROCK but no AWS_REGION.\n\nOpen this file:  {_env_file}\n"
        "and add the region your Bedrock models are enabled in, e.g.:\n"
        "    AWS_REGION=us-east-1\n\nSave the file, then click ▶ on this cell again."
    )


def _model(name):
    return f"anthropic.{name}" if PROVIDER == "bedrock" else name


MODEL = _model("claude-sonnet-5")
FAST_MODEL = _model("claude-haiku-4-5")
BIG_MODEL = _model("claude-opus-4-8")


def _make_client(timeout, max_retries=2):
    if PROVIDER == "bedrock":
        from anthropic import AnthropicBedrockMantle
        return AnthropicBedrockMantle(aws_region=_bedrock_region, timeout=timeout, max_retries=max_retries)
    return anthropic.Anthropic(api_key=_anthropic_key, timeout=timeout, max_retries=max_retries)


_probe = _make_client(timeout=30.0, max_retries=1)
try:
    _probe.messages.create(model=FAST_MODEL, max_tokens=1, messages=[{"role": "user", "content": "ping"}])
except anthropic.NotFoundError:
    if PROVIDER == "bedrock":
        _status(False, f"Bedrock reached, but model '{FAST_MODEL}' isn't available to you in "
                       f"{_bedrock_region}. Enable model access or switch AWS_REGION, then re-run.")
    else:
        _status(False, f"Model '{FAST_MODEL}' not found for this key.")
    raise SystemExit("Model not available — see the message above.")
except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
    if PROVIDER == "bedrock":
        _status(False, "That Bedrock key was rejected. Check AWS_BEARER_TOKEN_BEDROCK and permissions, then re-run.")
    else:
        _status(False, "That key was rejected. Run this cell again and paste the whole key (starts with sk-ant-).")
    raise SystemExit("Credentials not accepted - re-run this cell and try again.")
except Exception as exc:
    _status(False, "Could not reach the API (" + type(exc).__name__ + "). Check your connection, then run this cell again.")
    raise
else:
    if PROVIDER == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = _anthropic_key
        _status(True, "API key verified - you're connected to Claude.")
    else:
        _status(True, f"Bedrock key verified ({_bedrock_region}) - you're connected to Claude as {MODEL}.")

client = _make_client(timeout=900.0)

MODEL_HAIKU = FAST_MODEL
MODEL_SONNET = MODEL
MODEL_OPUS = BIG_MODEL

print(f"SDK {anthropic.__version__} · portfolio: {MODEL_HAIKU}, {MODEL_SONNET}, {MODEL_OPUS}")

# %% [markdown]
# # Part 1 · The scenario
#
# Everstock sells across three return-policy categories with materially different rules —
# this is what makes category handling, not just arithmetic, a real failure mode:
#
# | Category | Return window | Restocking fee | Final sale? |
# |---|---|---|---|
# | `electronics` | 30 days from delivery | 15% if opened/used, 0% if unopened | No |
# | `apparel` | 45 days from delivery | 0% | No |
# | `final_sale` | N/A | N/A | Yes — except a manufacturer defect claimed within 14 days |
#
# **Deliberate design choice:** the exact numbers above live *only* in `get_return_policy`'s
# tool output, never in the agent's system prompt. The system prompt (the cacheable
# "playbook") teaches the *decision framework* — when to escalate, what counts as fraud
# signal, the discipline never to invent an exception — the same split ClauseScan used
# between its playbook (methodology) and each contract's text (case facts).

# %%
TODAY = date(2026, 9, 18)  # fixed review date so day-count math never drifts between runs

RETURN_POLICY_TEXT = {
    "electronics": {
        "return_window_days": 30,
        "restocking_fee_pct_if_opened": 15,
        "final_sale": False,
        "policy_text": (
            "ELECTRONICS RETURN POLICY: Returns accepted within 30 days of delivery. "
            "Unopened or unused items are refunded in full. Opened or used items incur a 15% "
            "restocking fee. A verified manufacturer defect is exempt from the restocking fee "
            "regardless of condition. Electronics are not eligible for return through Everstock "
            "after 30 days for any reason; a defect claimed after 30 days is handled by the "
            "manufacturer's own warranty, not by Everstock."
        ),
    },
    "apparel": {
        "return_window_days": 45,
        "restocking_fee_pct_if_opened": 0,
        "final_sale": False,
        "policy_text": (
            "APPAREL RETURN POLICY: Returns accepted within 45 days of delivery, free of "
            "charge, with no restocking fee. Items must be unworn and unwashed unless the "
            "return reason is a manufacturing defect. Worn, altered, or damaged apparel is "
            "denied unless a defect is verified. There is no loyalty-tier extension, grace "
            "period, or manager override of the 45-day window."
        ),
    },
    "final_sale": {
        "return_window_days": None,
        "restocking_fee_pct_if_opened": None,
        "final_sale": True,
        "policy_text": (
            "FINAL SALE POLICY: Items marked final_sale are not eligible for return or refund "
            "for any reason, including size, color, fit, or change of mind — regardless of "
            "whether the item was opened. The sole exception is a manufacturer defect claimed "
            "within 14 days of delivery. There is no second exception: no loyalty-tier "
            "override, no manager discretion, no goodwill extension, no grace period."
        ),
    },
}

# Orders: 8 map directly to the eval scenarios in Part 4, 2 are holdout (Part 6), 2 are
# unlabeled demo orders used only in Part 3's live trace.
ORDERS = {
    # -- demo orders (Part 3) --
    "ORD-3001": {"order_id": "ORD-3001", "customer_id": "CUST-5001", "item": "Espresso Machine",
                 "category": "electronics", "price_usd": 349.00, "purchase_date": "2026-09-08",
                 "delivery_date": "2026-09-13", "order_status": "delivered", "opened": False},
    "ORD-3002": {"order_id": "ORD-3002", "customer_id": "CUST-5002", "item": "Denim Jacket",
                 "category": "apparel", "price_usd": 68.00, "purchase_date": "2026-08-14",
                 "delivery_date": "2026-08-19", "order_status": "delivered", "opened": True},
    # -- in-sample eval orders (Part 4) --
    "ORD-2001": {"order_id": "ORD-2001", "customer_id": "CUST-4001", "item": "Merino Wool Sweater",
                 "category": "apparel", "price_usd": 79.00, "purchase_date": "2026-08-24",
                 "delivery_date": "2026-08-29", "order_status": "delivered", "opened": True},
    "ORD-2002": {"order_id": "ORD-2002", "customer_id": "CUST-4002", "item": "Clearance Running Shoes",
                 "category": "final_sale", "price_usd": 45.00, "purchase_date": "2026-09-03",
                 "delivery_date": "2026-09-08", "order_status": "delivered", "opened": True},
    "ORD-2003": {"order_id": "ORD-2003", "customer_id": "CUST-4003", "item": "4K Action Camera",
                 "category": "electronics", "price_usd": 199.00, "purchase_date": "2026-09-01",
                 "delivery_date": "2026-09-06", "order_status": "delivered", "opened": True},
    "ORD-2004": {"order_id": "ORD-2004", "customer_id": "CUST-4004", "item": "Bluetooth Speaker",
                 "category": "electronics", "price_usd": 89.00, "purchase_date": "2026-07-15",
                 "delivery_date": "2026-07-20", "order_status": "delivered", "opened": True},
    "ORD-2005": {"order_id": "ORD-2005", "customer_id": "CUST-4005", "item": "Bluetooth Earbuds",
                 "category": "electronics", "price_usd": 129.00, "purchase_date": "2026-08-29",
                 "delivery_date": "2026-09-03", "order_status": "delivered", "opened": True},
    "ORD-2006": {"order_id": "ORD-2006", "customer_id": "CUST-4006", "item": "Limited Edition Sneakers (Clearance)",
                 "category": "final_sale", "price_usd": 150.00, "purchase_date": "2026-09-08",
                 "delivery_date": "2026-09-13", "order_status": "delivered", "opened": False},
    "ORD-2007": {"order_id": "ORD-2007", "customer_id": "CUST-4007", "item": "Professional Camera Kit",
                 "category": "electronics", "price_usd": 2499.00, "purchase_date": "2026-09-05",
                 "delivery_date": "2026-09-10", "order_status": "delivered", "opened": True},
    "ORD-2008": {"order_id": "ORD-2008", "customer_id": "CUST-4008", "item": "Wool Peacoat",
                 "category": "apparel", "price_usd": 189.00, "purchase_date": "2026-07-19",
                 "delivery_date": "2026-07-24", "order_status": "delivered", "opened": True},
    # -- holdout orders (Part 6) --
    "ORD-2101": {"order_id": "ORD-2101", "customer_id": "CUST-4101", "item": "Yoga Leggings",
                 "category": "apparel", "price_usd": 54.00, "purchase_date": "2026-08-29",
                 "delivery_date": "2026-09-03", "order_status": "delivered", "opened": False},
    "ORD-2102": {"order_id": "ORD-2102", "customer_id": "CUST-4102", "item": "Cocktail Dress",
                 "category": "apparel", "price_usd": 120.00, "purchase_date": "2026-09-05",
                 "delivery_date": "2026-09-10", "order_status": "delivered", "opened": True},
}

CUSTOMER_RETURN_HISTORY = {
    "CUST-5001": {"total_orders_365d": 6, "return_count_90d": 0, "return_count_365d": 1, "abuse_flags": [], "chargeback_count": 0},
    "CUST-5002": {"total_orders_365d": 9, "return_count_90d": 1, "return_count_365d": 2, "abuse_flags": [], "chargeback_count": 0},
    "CUST-4001": {"total_orders_365d": 4, "return_count_90d": 0, "return_count_365d": 0, "abuse_flags": [], "chargeback_count": 0},
    "CUST-4002": {"total_orders_365d": 2, "return_count_90d": 0, "return_count_365d": 1, "abuse_flags": [], "chargeback_count": 0},
    "CUST-4003": {"total_orders_365d": 5, "return_count_90d": 1, "return_count_365d": 1, "abuse_flags": [], "chargeback_count": 0},
    "CUST-4004": {"total_orders_365d": 3, "return_count_90d": 0, "return_count_365d": 0, "abuse_flags": [], "chargeback_count": 0},
    "CUST-4005": {"total_orders_365d": 11, "return_count_90d": 4, "return_count_365d": 9,
                  "abuse_flags": ["serial_full_price_returner"], "chargeback_count": 1},
    "CUST-4006": {"total_orders_365d": 7, "return_count_90d": 0, "return_count_365d": 1, "abuse_flags": [], "chargeback_count": 0},
    "CUST-4007": {"total_orders_365d": 3, "return_count_90d": 0, "return_count_365d": 0, "abuse_flags": [], "chargeback_count": 0},
    "CUST-4008": {"total_orders_365d": 8, "return_count_90d": 1, "return_count_365d": 2, "abuse_flags": [], "chargeback_count": 0},
    "CUST-4101": {"total_orders_365d": 5, "return_count_90d": 0, "return_count_365d": 0, "abuse_flags": [], "chargeback_count": 0},
    "CUST-4102": {"total_orders_365d": 10, "return_count_90d": 3, "return_count_365d": 6,
                  "abuse_flags": ["wardrobing_pattern"], "chargeback_count": 0},
}

print(f"Loaded {len(ORDERS)} orders, {len(CUSTOMER_RETURN_HISTORY)} customer histories, "
      f"{len(RETURN_POLICY_TEXT)} policy categories")

# %% [markdown]
# ### The playbook — the cacheable, static decision framework
#
# Notice what's *not* here: no return-window day counts, no restocking-fee percentages, no
# per-order facts. Those only exist behind the tools. That split is what makes "call
# `get_return_policy` every time, even for a category you just handled" a real instruction
# rather than a formality — and it's what makes the hallucination-trap eval case in Part 4
# meaningful.

# %%
PLAYBOOK_CORE = """You are the Returns & Refund Decisioning Agent for Everstock, a mid-size \
e-commerce retailer that processes thousands of return requests per day. For each request you \
reach exactly one of four decisions and justify it only with facts you looked up this turn — \
never recall or assume a policy detail, price, date, or history fact from memory or from a \
prior case.

## Decisions
- APPROVE — full refund, no restocking fee withheld.
- PARTIAL_REFUND — refund minus a restocking fee (only where the category's policy defines one).
- DENY — no refund.
- ESCALATE_TO_HUMAN — you cannot responsibly decide without a person's judgment.

## Required tool calls, every single time
1. Call get_order first. Never assume an item's price, category, or dates from the customer's \
message — customers misdescribe or omit details.
2. Call get_return_policy for the order's category on every request, even for a category you \
handled a moment ago. Policy terms are not fixed in your instructions and can change; your \
policy_citation in the final decision must be an exact or near-exact quote from that tool's \
output, never from memory.
3. Call get_customer_return_history whenever the order's price_usd is above $150, the category \
is electronics, or the case is otherwise borderline. Fraud and abuse risk cannot be judged from \
the order alone.

## Escalation rules
- If get_return_policy's response does not mention an exception the customer is asking about (a \
loyalty-tier grace period, a manager override, a goodwill extension, or anything else), that \
exception does not exist. State plainly that no such exception applies — do not invent one to \
be accommodating.
- A shipping error (wrong item shipped, item defective on arrival, damaged in transit) is \
Everstock's fault. Treat it as a service-recovery case: approve it regardless of whether the \
normal return window has passed.
- If the customer's return history shows any abuse flag, or 3 or more returns in the trailing \
90 days, escalate to a human — even if the item itself would otherwise qualify for approval on \
its own terms.
- If a single item's price_usd is above $500 and the condition or defect claim is ambiguous or \
disputed (not a clean "unopened" or a clean, undisputed defect description), escalate to a \
human rather than guessing at the restocking-fee math.
- Order records include days_since_delivery, already computed as of today's review date — use \
that number directly. Do not compute your own date arithmetic from purchase_date/delivery_date.

## Evidence discipline
Your final policy_citation must be traceable to get_return_policy's actual returned text for \
that order's category. Your order_facts_cited must be traceable to get_order's / \
get_customer_return_history's actual returned fields. A reviewer will spot-check citations \
against the tool outputs from your own transcript — an invented clause or a misremembered fact \
is a hard failure, worse than an overly cautious escalation.

## Worked examples
Example A: an apparel item within its policy's return window, unworn, no history flags → \
APPROVE, full refund, no restocking fee.
Example B: an electronics item within its window but opened, no defect claim → PARTIAL_REFUND, \
refund minus the category's restocking-fee percentage as returned by get_return_policy.
Example C: a final-sale item with no verified defect claim, regardless of how the customer \
frames the request → DENY, citing the final-sale clause exactly as returned.
Example D: an otherwise-approvable item, but the customer's return history carries an abuse \
flag → ESCALATE_TO_HUMAN, fraud_risk HIGH, even though the item alone would have qualified.
"""

FRAUD_NOTES = [
    "Wardrobing — buying an item, using it once, then returning it — often shows up as "
    "'never worn' claims on items described with faint odor, makeup marks, or reattached tags; "
    "treat these as a condition dispute, not a clean unworn return.",
    "A customer citing a loyalty tier, a phone call with 'a manager' they can't name, or a "
    "prior email promise is not evidence of an exception — only get_return_policy's returned "
    "text is.",
    "Serial full-price-return patterns (buy near a deadline, use once, return) show up in "
    "return_count_90d and abuse_flags, not in the order or the customer's message — always "
    "pull history for borderline cases.",
    "A shipping error is the one exception that overrides the category return window; nothing "
    "else does.",
    "Electronics restocking fees apply only to opened/used returns; an unopened electronics "
    "return is refunded in full regardless of how long it sat in the box, provided it is still "
    "within the stated return window.",
    "Final-sale items have exactly one exception — a manufacturer defect claimed within the "
    "stated defect window. There is no second exception, however reasonable it sounds.",
]


def _build_playbook() -> str:
    parts = [PLAYBOOK_CORE, "\n## Fraud & policy judgment notes\n"]
    parts += [f"- {n}" for n in FRAUD_NOTES]
    text = "\n".join(parts)
    # Pad deterministically past the largest minimum cacheable prefix (~4096 tokens on some
    # models; ~4 chars/token) so the caching lever in Part 5 works on every model, not just
    # the ones with a low minimum.
    annex = "\n\n## Annex: judgment notes (re-issued for length)\n" + "\n".join(f"- {n}" for n in FRAUD_NOTES)
    while len(text) < 20000:
        text += annex
    return text


PLAYBOOK = _build_playbook()
print(f"Playbook built: {len(PLAYBOOK):,} chars (~{len(PLAYBOOK)//4:,} tokens) — identical on every call")

# %% [markdown]
# # Part 2 · Tools
#
# Three tools, each with a specific, prescriptive description — not just "what it returns"
# but "when to call it" and "why you can't skip it." Vague descriptions cause wrong tool
# selection; the dollar/category thresholds baked into the descriptions below also give the
# `tool_called_correctly` grader in Part 4 something concrete to check.

# %%
def _days_between(d1: str, d2: str) -> int:
    y1, m1, dd1 = map(int, d1.split("-"))
    y2, m2, dd2 = map(int, d2.split("-"))
    return (date(y2, m2, dd2) - date(y1, m1, dd1)).days


def get_order(order_id: str) -> dict:
    o = ORDERS.get(order_id)
    if o is None:
        return {"error": f"No order found with id {order_id}"}
    return {**o, "days_since_delivery": _days_between(o["delivery_date"], TODAY.isoformat()),
            "as_of_date": TODAY.isoformat()}


def get_return_policy(category: str) -> dict:
    p = RETURN_POLICY_TEXT.get(category)
    if p is None:
        return {"error": f"Unknown category {category}"}
    return dict(p)


def get_customer_return_history(customer_id: str) -> dict:
    h = CUSTOMER_RETURN_HISTORY.get(customer_id)
    if h is None:
        return {"error": f"No history found for customer {customer_id}"}
    return {"customer_id": customer_id, **h}


TOOL_REGISTRY = {
    "get_order": get_order,
    "get_return_policy": get_return_policy,
    "get_customer_return_history": get_customer_return_history,
}

GET_ORDER_SPEC = {
    "name": "get_order",
    "description": (
        "Retrieve the full details of a single Everstock order by its order ID: item name, "
        "product category (electronics, apparel, or final_sale), price paid, purchase date, "
        "delivery date, order status, and days_since_delivery (already computed as of today's "
        "review date — use it directly). Always call this first for any return request — never "
        "assume the item's category, price, or purchase date from the customer's message; the "
        "customer's description may be wrong or incomplete."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "The order ID, e.g. 'ORD-2001'."}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
}

GET_RETURN_POLICY_SPEC = {
    "name": "get_return_policy",
    "description": (
        "Look up Everstock's official written return policy for one product category. Returns "
        "the return window in days, the restocking fee percentage, whether the category is "
        "final-sale, and the exact policy clause text to cite as evidence. Call this for every "
        "decision, even if you believe you already know the category's policy — policy terms "
        "vary by category and can change, and your decision's evidence must quote this tool's "
        "output, not a memorized or assumed rule."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": ["electronics", "apparel", "final_sale"],
                         "description": "The product category returned by get_order for this order's item."}
        },
        "required": ["category"],
        "additionalProperties": False,
    },
}

GET_CUSTOMER_HISTORY_SPEC = {
    "name": "get_customer_return_history",
    "description": (
        "Retrieve a customer's return behavior over the trailing 90 and 365 days: total orders, "
        "return counts, any automated abuse-pattern flags (e.g. 'serial_full_price_returner', "
        "'wardrobing_pattern'), and chargeback count. Call this before deciding on any order "
        "priced above $150, any electronics return, or any case that is otherwise borderline — "
        "fraud risk cannot be assessed from the order alone."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"customer_id": {"type": "string", "description": "The customer ID from the order record, e.g. 'CUST-4001'."}},
        "required": ["customer_id"],
        "additionalProperties": False,
    },
}

TOOL_SPECS = [GET_ORDER_SPEC, GET_RETURN_POLICY_SPEC, GET_CUSTOMER_HISTORY_SPEC]


def execute_tool(name: str, inputs: dict) -> str:
    try:
        return json.dumps(TOOL_REGISTRY[name](**inputs))
    except Exception as e:
        return json.dumps({"error": str(e)})


print(f"Tools ready: {list(TOOL_REGISTRY)}")

# %% [markdown]
# ### The structured decision schema
#
# Used only on the *final* call, after the tool loop exits — the same sequencing the
# Developer Platform module used: run the loop with `tools` available and no
# `output_config`; once `stop_reason != "tool_use"`, make one more call with
# `tool_choice: {"type": "none"}` plus `output_config.format` for the structured decision.
# `policy_citation` and `order_facts_cited` are the load-bearing fields — they give the
# Part 4 hallucination grader something concrete to cross-check against the actual tool
# outputs from that run's transcript, rather than judging prose in isolation.

# %%
DECISION_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["APPROVE", "PARTIAL_REFUND", "DENY", "ESCALATE_TO_HUMAN"]},
            "refund_amount_usd": {"anyOf": [{"type": "number"}, {"type": "null"}],
                                  "description": "Dollar amount to refund. null for DENY or ESCALATE_TO_HUMAN."},
            "restocking_fee_applied_usd": {"anyOf": [{"type": "number"}, {"type": "null"}],
                                           "description": "Dollar amount withheld as a restocking fee. null if none applies."},
            "fraud_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
            "policy_citation": {"type": "string",
                                "description": "The exact policy clause text from get_return_policy's output that this decision relies on. A real quote, not a paraphrase or invented rule."},
            "order_facts_cited": {"type": "array", "items": {"type": "string"},
                                  "description": "The specific order/history facts relied on, e.g. 'delivered 2026-08-29, 20 days ago, category apparel'."},
            "reasoning_summary": {"type": "string", "description": "One or two sentences explaining the decision."},
        },
        "required": ["decision", "refund_amount_usd", "restocking_fee_applied_usd", "fraud_risk",
                     "policy_citation", "order_facts_cited", "reasoning_summary"],
        "additionalProperties": False,
    },
}

print("Decision schema ready")

# %% [markdown]
# # Part 3 · The agentic loop
#
# `run_tool_loop` is the same `while stop_reason == "tool_use"` shape from the Developer
# Platform module, extended to collect every tool call (with its arguments *and* result) for
# later grading. `collect_tool_calls` and `final_decision_call` are the two halves of the
# "loop first, structured decision last" split described above.

# %%
def run_tool_loop(query: str, model: str, system):
    messages = [{"role": "user", "content": query}]
    calls = []
    while True:
        resp = client.messages.create(model=model, max_tokens=1200, system=system,
                                      tools=TOOL_SPECS, messages=messages)
        calls.append((model, resp.usage))
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        tool_uses = [b for b in resp.content if isinstance(b, ToolUseBlock)]
        results = [{"type": "tool_result", "tool_use_id": tu.id, "content": execute_tool(tu.name, tu.input)}
                   for tu in tool_uses]
        messages.append({"role": "user", "content": results})
    return messages, calls


def collect_tool_calls(messages) -> list:
    """Extract name/arguments/result triples from a transcript — used by graders and the
    LLM judge to check what the agent actually saw, not what it claims to have seen."""
    tool_calls = []
    for msg in messages:
        if msg["role"] == "assistant":
            for b in msg["content"]:
                if isinstance(b, ToolUseBlock):
                    tool_calls.append({"name": b.name, "arguments": b.input, "id": b.id})
        elif msg["role"] == "user" and isinstance(msg["content"], list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    for tc in tool_calls:
                        if tc["id"] == item["tool_use_id"]:
                            tc["result"] = item.get("content", "")
    return tool_calls


def text_of(response) -> str:
    return "".join(b.text for b in response.content if isinstance(b, TextBlock))


def extract_json(text: str):
    """Pull the first valid JSON object out of free text (v0's prose pass needs this)."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        break
    return None


def final_decision_call(messages, model: str, system, effort: Optional[str] = None):
    """messages, as returned by run_tool_loop, always ends on an assistant turn (the loop
    appends the assistant's message immediately before checking stop_reason). Current-gen
    models reject a request whose conversation ends on 'assistant' as unsupported prefill —
    so we append one synthetic user turn asking for the decision, locally, without mutating
    the caller's messages list (collect_tool_calls doesn't need to see it)."""
    output_config = {"format": DECISION_SCHEMA}
    if effort and model != MODEL_HAIKU:  # effort errors on Haiku 4.5
        output_config["effort"] = effort
    request_messages = messages + [{"role": "user", "content":
        "Based on everything you've found, provide your final decision now, matching the required schema."}]
    resp = client.messages.create(model=model, max_tokens=1200, system=system, tools=TOOL_SPECS,
                                  tool_choice={"type": "none"}, messages=request_messages, output_config=output_config)
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("final_decision_call hit max_tokens before completing the schema — raise max_tokens")
    return json.loads(text_of(resp)), resp


def decide_efficient(query: str, model: str, system, effort: Optional[str] = None) -> dict:
    """The efficient shape used everywhere except the deliberately wasteful v0: one tool
    loop, one structured decision call."""
    t0 = time.perf_counter()
    messages, calls = run_tool_loop(query, model, system)
    decision, final_resp = final_decision_call(messages, model, system, effort=effort)
    calls.append((model, final_resp.usage))
    return {"decision": decision, "tool_calls": collect_tool_calls(messages), "calls": calls,
            "elapsed": time.perf_counter() - t0, "messages": messages}


print("Agentic loop ready")

# %% [markdown]
# ### Try it live
#
# Two orders never used in the Part 4 eval set, so this trace doesn't leak an answer key:
# an easy unopened-electronics approval, and a worn-apparel denial. Read the printed trace —
# which tools fired, in what order, with what arguments — before trusting anything below it.

# %%
def print_trace(result: dict, label: str):
    print(f"\n── {label} " + "─" * max(1, 60 - len(label)))
    for c in result["tool_calls"]:
        print(f"  [tool_use] {c['name']}({c['arguments']}) -> {c.get('result', '')[:160]}")
    print(f"  [decision] {json.dumps(result['decision'], indent=2)}")
    print(f"  {len(result['calls'])} API calls · {result['elapsed']:.1f}s")


demo1 = decide_efficient("Hi, I'd like to return order ORD-3001 — I changed my mind, box is unopened.",
                         model=MODEL_SONNET, system=PLAYBOOK)
print_trace(demo1, "ORD-3001 — unopened electronics, change of mind")

demo2 = decide_efficient("Returning order ORD-3002, the jacket just doesn't look right on me. I've worn it a few times.",
                         model=MODEL_SONNET, system=PLAYBOOK)
print_trace(demo2, "ORD-3002 — worn apparel, no defect claim")

# %% [markdown]
# # Part 4 · Naive v0 — the inherited baseline
#
# `everstock_v0`: one expensive model for everything, no caching, and — after the tool loop
# gathers facts — a second, unconstrained pass that free-writes a reasoning essay and then a
# JSON blob at the end, regex-extracted. No schema means no guarantee the citation is a real
# quote rather than something that merely *sounds* like policy. That gap is exactly what the
# hallucination grader below is built to catch.

# %%
def everstock_v0(task: dict) -> dict:
    t0 = time.perf_counter()
    messages, calls = run_tool_loop(task["query"], MODEL_OPUS, PLAYBOOK)
    messages.append({"role": "user", "content": (
        "Now explain your reasoning step by step in detail, weighing all four decision types, "
        "and then output a JSON object with keys decision, refund_amount_usd, "
        "restocking_fee_applied_usd, fraud_risk, policy_citation, order_facts_cited, "
        "reasoning_summary."
    )})
    resp = client.messages.create(model=MODEL_OPUS, max_tokens=3000, system=PLAYBOOK, messages=messages)
    calls.append((MODEL_OPUS, resp.usage))
    messages.append({"role": "assistant", "content": resp.content})
    decision = extract_json(text_of(resp))
    if decision is None and resp.stop_reason == "max_tokens":
        raise RuntimeError("everstock_v0 hit max_tokens before completing its essay+JSON pass — raise max_tokens")
    return {"decision": decision, "tool_calls": collect_tool_calls(messages), "calls": calls,
            "elapsed": time.perf_counter() - t0, "messages": messages}


print("ClauseScan-style v0 loaded — 8 in-sample scenarios follow in Part 5's eval run")

# %% [markdown]
# # Part 5 · The eval harness
#
# From scratch, mirroring the eval-building session: a `tasks` list, a `GRADER_REGISTRY`
# dict (not if/elif), and a runner. One synthesis beyond that session's version: the runner
# here also tracks cost and latency per task (cache-aware `calculate_cost`, the same
# formula from the optimization lab), so the *same* harness produces both pass/fail grading
# **and** the accuracy-gated scorecard used in Part 7 — no second, separate measurement
# system needed.
#
# Eight in-sample scenarios, one per required failure mode, plus two holdout scenarios held
# back until Part 7. `hallucination_trap` is the anchor case: the customer asks about a
# loyalty-tier grace period that does not exist anywhere in Everstock's actual policy.

# %%
PRICING = {
    MODEL_HAIKU: {"input": 1.00, "output": 5.00},
    MODEL_SONNET: {"input": 3.00, "output": 15.00},
    MODEL_OPUS: {"input": 5.00, "output": 25.00},
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10


def calculate_cost(model: str, usage) -> float:
    p = PRICING[model]
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (usage.input_tokens * p["input"] + cache_write * p["input"] * CACHE_WRITE_MULT
            + cache_read * p["input"] * CACHE_READ_MULT + usage.output_tokens * p["output"]) / 1e6


JUDGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
            "reason": {"type": "string", "description": "One sentence, naming the specific evidence."},
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    },
}


def grade_decision_match(result, check, context=None):
    got = (result["decision"] or {}).get("decision")
    ok = str(got or "").upper() == str(check).upper()
    return {"score": 1.0 if ok else 0.0, "reason": f"decision={got!r} expected={check!r}"}


def grade_refund_within_tolerance(result, check, context=None):
    expected = check.get("value") if isinstance(check, dict) else check
    tolerance = check.get("tolerance", 0.01) if isinstance(check, dict) else 0.01
    got = (result["decision"] or {}).get("refund_amount_usd")
    if expected is None:
        ok = got is None
        return {"score": 1.0 if ok else 0.0, "reason": f"expected null refund, got {got}"}
    if got is None:
        return {"score": 0.0, "reason": f"expected refund ~{expected}, got null"}
    ok = abs(float(got) - float(expected)) <= tolerance
    return {"score": 1.0 if ok else 0.0, "reason": f"refund={got} expected {expected}+/-{tolerance}"}


def grade_tool_called(result, check, context=None):
    tool_name = check["tool_name"]
    called = any(c["name"] == tool_name for c in result["tool_calls"])
    actual = [c["name"] for c in result["tool_calls"]]
    return {"score": 1.0 if called else 0.0, "reason": f"{tool_name} called={called}; actual={actual}"}


def grade_no_hallucination(result, check, context=None):
    decision = result["decision"] or {}
    evidence = "\n".join(f"- {c['name']}({c['arguments']}) -> {c.get('result', '')}" for c in result["tool_calls"])
    judge_prompt = (
        "You are auditing an e-commerce returns agent for hallucination. Below are the ACTUAL "
        "tool outputs it saw this turn (ground truth), followed by its final decision.\n\n"
        f"TOOL OUTPUTS:\n{evidence}\n\n"
        f"FINAL DECISION:\n"
        f"decision: {decision.get('decision')}\n"
        f"policy_citation: {decision.get('policy_citation')}\n"
        f"order_facts_cited: {decision.get('order_facts_cited')}\n"
        f"reasoning_summary: {decision.get('reasoning_summary')}\n\n"
        f"Criterion to check: {check}\n\n"
        "Does the final decision satisfy this criterion? Judge only against the tool outputs "
        "above — do not use any outside knowledge of what a 'reasonable' return policy might say."
    )
    resp = client.messages.create(model=MODEL_HAIKU, max_tokens=300,
                                  messages=[{"role": "user", "content": judge_prompt}],
                                  output_config={"format": JUDGE_SCHEMA})
    data = json.loads(text_of(resp))
    return {"score": 1.0 if data["verdict"] == "PASS" else 0.0, "reason": data["reason"]}


GRADER_REGISTRY = {
    "decision_match": grade_decision_match,
    "refund_within_tolerance": grade_refund_within_tolerance,
    "tool_called": grade_tool_called,
    "no_hallucination": grade_no_hallucination,
}

print(f"Graders loaded: {list(GRADER_REGISTRY)}")

# %%
TASKS = [
    {"id": "clean_approve_apparel", "order_id": "ORD-2001",
     "query": "Hi, I'd like to return order ORD-2001 — the sweater just doesn't fit right, and I haven't worn it.",
     "graders": [
         {"type": "decision_match", "checks": ["APPROVE"]},
         {"type": "refund_within_tolerance", "checks": [{"value": 79.00, "tolerance": 0.01}]},
     ]},
    {"id": "clean_deny_final_sale", "order_id": "ORD-2002",
     "query": "I need to return order ORD-2002, the shoes are the wrong size for me.",
     "graders": [
         {"type": "decision_match", "checks": ["DENY"]},
         {"type": "refund_within_tolerance", "checks": [{"value": None}]},
     ]},
    {"id": "partial_refund_restocking", "order_id": "ORD-2003",
     "query": "Returning order ORD-2003, the action camera. I opened it and used it a couple times but changed my mind.",
     "graders": [
         {"type": "decision_match", "checks": ["PARTIAL_REFUND"]},
         {"type": "refund_within_tolerance", "checks": [{"value": 169.15, "tolerance": 0.05}]},
     ]},
    {"id": "past_window_shipping_error", "order_id": "ORD-2004",
     "query": "Order ORD-2004 — I know it's been a while, but you shipped me the wrong Bluetooth speaker model entirely. I want to return it.",
     "graders": [
         {"type": "decision_match", "checks": ["APPROVE"]},
         {"type": "refund_within_tolerance", "checks": [{"value": 89.00, "tolerance": 0.01}]},
     ]},
    {"id": "serial_returner_fraud", "order_id": "ORD-2005",
     "query": "I'd like to return order ORD-2005, the earbuds. Just don't like the sound quality.",
     "graders": [
         {"type": "decision_match", "checks": ["ESCALATE_TO_HUMAN"]},
         {"type": "tool_called", "checks": [{"tool_name": "get_customer_return_history"}]},
     ]},
    {"id": "final_sale_claimed_otherwise", "order_id": "ORD-2006",
     "query": "Order ORD-2006 — I know it says final sale, but I never even opened the box, so surely that still counts as returnable?",
     "graders": [
         {"type": "decision_match", "checks": ["DENY"]},
         {"type": "no_hallucination", "checks": [
             "The decision does not grant an 'unopened items are always returnable' exception "
             "for a final-sale item, since no such exception exists in the actual final_sale policy text."]},
     ]},
    {"id": "high_value_escalation", "order_id": "ORD-2007",
     "query": "Returning order ORD-2007, the camera kit. It arrived with a scratch on the lens — not totally sure if that happened in shipping or if it's a defect.",
     "graders": [
         {"type": "decision_match", "checks": ["ESCALATE_TO_HUMAN"]},
         {"type": "tool_called", "checks": [{"tool_name": "get_customer_return_history"}]},
     ]},
    {"id": "hallucination_trap", "order_id": "ORD-2008",
     "query": "I need to return order ORD-2008. I know it's been a couple months, but I'm a Gold loyalty member — don't I get a grace period on the return window?",
     "graders": [
         {"type": "decision_match", "checks": ["DENY"]},
         {"type": "no_hallucination", "checks": [
             "The decision does not reference or grant any loyalty-tier grace period, "
             "membership extension, or return-window exception, since Everstock's actual "
             "apparel policy text contains no such exception."]},
     ]},
]

HOLDOUT_TASKS = [
    {"id": "holdout_clean", "order_id": "ORD-2101",
     "query": "I'd like to return order ORD-2101, the leggings. They're the wrong size, never worn.",
     "graders": [{"type": "decision_match", "checks": ["APPROVE"]},
                 {"type": "refund_within_tolerance", "checks": [{"value": 54.00, "tolerance": 0.01}]}]},
    {"id": "holdout_edge", "order_id": "ORD-2102",
     "query": "Returning order ORD-2102, the dress. It just wasn't what I expected.",
     "graders": [{"type": "decision_match", "checks": ["ESCALATE_TO_HUMAN"]},
                 {"type": "tool_called", "checks": [{"tool_name": "get_customer_return_history"}]}]},
]

print(f"{len(TASKS)} in-sample tasks, {len(HOLDOUT_TASKS)} holdout tasks")

# %% [markdown]
# ### The runner
#
# `run_batch` grades every check for every task (`GRADER_REGISTRY`-style) *and* aggregates
# cost/p50/accuracy in the same pass (`run_portfolio`-style) — one harness serving both
# purposes described above.

# %%
def run_batch(pipeline, tasks, workers: int = 0, warm_first: bool = True) -> dict:
    def run_one(task):
        out = pipeline(task)
        grades = []
        result_for_grading = {"decision": out["decision"], "tool_calls": out["tool_calls"]}
        for grader in task.get("graders", []):
            fn = GRADER_REGISTRY[grader["type"]]
            for check in grader["checks"]:
                try:
                    g = fn(result_for_grading, check, {"task_id": task["id"]})
                except Exception as exc:
                    g = {"score": 0.0, "reason": f"grader error: {type(exc).__name__}: {exc}"}
                grades.append({"type": grader["type"], "check": check, **g})
        passed = all(g["score"] == 1.0 for g in grades) if grades else False
        cost = sum(calculate_cost(m, u) for m, u in out["calls"])
        return {"id": task["id"], "decision": out["decision"], "grades": grades, "passed": passed,
                "n_correct": sum(g["score"] for g in grades), "n_checks": len(grades),
                "elapsed": out["elapsed"], "cost": cost, "calls": len(out["calls"])}

    wall0 = time.perf_counter()
    if workers and len(tasks) > 1:
        rows = []
        head, remaining = (tasks[0], tasks[1:]) if warm_first else (None, tasks)
        if head is not None:
            rows.append(run_one(head))  # one call writes the cache before the fan-out
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows.extend(ex.map(run_one, remaining))
    else:
        rows = [run_one(t) for t in tasks]
    wall = time.perf_counter() - wall0

    order = {t["id"]: i for i, t in enumerate(tasks)}
    rows.sort(key=lambda r: order.get(r["id"], 999))
    total_checks = sum(r["n_checks"] for r in rows)
    return {"rows": rows, "accuracy": sum(r["n_correct"] for r in rows) / max(total_checks, 1),
            "task_pass_rate": sum(r["passed"] for r in rows) / len(rows),
            "p50_s": statistics.median(r["elapsed"] for r in rows),
            "cost_per_decision": sum(r["cost"] for r in rows) / len(rows),
            "total_cost": sum(r["cost"] for r in rows), "wall_s": wall}


def print_report(report: dict, label: str):
    rows = [[r["id"], "PASS" if r["passed"] else "FAIL", f"{int(r['n_correct'])}/{r['n_checks']}",
             f"{r['elapsed']:.1f}s", f"${r['cost']:.4f}", r["calls"]] for r in report["rows"]]
    print(f"\n── {label} " + "─" * max(1, 60 - len(label)))
    print(tabulate(rows, headers=["ID", "Task", "Checks", "TTC", "Cost", "Calls"], tablefmt="simple"))
    for r in report["rows"]:
        if not r["passed"]:
            for g in r["grades"]:
                if g["score"] != 1.0:
                    print(f"    [{r['id']}] - {g['type']}: {g['reason'][:150]}")
    print(f"\n  accuracy {report['accuracy']*100:.0f}%  ·  task pass rate {report['task_pass_rate']*100:.0f}%  ·  "
          f"p50 {report['p50_s']:.1f}s/decision  ·  ${report['cost_per_decision']:.4f}/decision  ·  "
          f"wall-clock {report['wall_s']:.0f}s")


print("Running Everstock v0 on the 8 in-sample scenarios (sequential, Opus, uncached — "
      "the slowness and cost ARE the data)...")
BASELINE = run_batch(everstock_v0, TASKS)
print_report(BASELINE, "Everstock v0 — the naive baseline")

# %% [markdown]
# ### Is this eval trustworthy?
#
# Before building anything on top of this harness: does it actually fail v0 anywhere, or is
# it a rubber stamp? Check the printed failures above — `hallucination_trap` failing with a
# reason that names the invented loyalty exception (not a generic "FAIL") is the one that
# matters most. If v0 passed everything, the dataset isn't hard enough yet.
#
# One more check worth running by hand: take a transcript that *passed* `no_hallucination`,
# hand-mutate its `policy_citation` to something fabricated, and re-run just that grader —
# confirm it flips to FAIL. A judge that can't catch an injected mutation isn't trustworthy
# for anything downstream.

# %% [markdown]
# # Part 6 · Optimization levers
#
# Same five levers from the inference-optimization lab, applied here. Each is shown in
# isolation on a small, cheap demo rather than re-running the full 8-task set five times —
# lever 5 (parallelism) is demonstrated directly in the Part 7 sprint instead of here, so the
# full task set only gets run twice more (sequential vs. parallel `v1`), not five times.

# %% [markdown]
# ## Lever 1 — Prompt caching

# %%
CACHED_SYSTEM = [{"type": "text", "text": PLAYBOOK, "cache_control": {"type": "ephemeral"}}]

for label in ("COLD (writes cache)", "WARM (reads cache)"):
    resp = client.messages.create(model=MODEL_SONNET, max_tokens=200, system=CACHED_SYSTEM,
                                  tools=TOOL_SPECS,
                                  messages=[{"role": "user", "content": TASKS[0]["query"]}])
    u = resp.usage
    print(f"{label:20s} uncached_in={u.input_tokens:>5} · cache_write={u.cache_creation_input_tokens or 0:>5} · "
          f"cache_read={u.cache_read_input_tokens or 0:>5} · cost=${calculate_cost(MODEL_SONNET, u):.5f}")

print("\nWatch cache_read jump from 0 to ~the playbook size on the second call.")

# %% [markdown]
# ## Lever 2 — Model routing: a portfolio, not one model

# %%
TRIAGE_SYSTEM = (
    "You triage e-commerce return requests for a decisioning pipeline. Classify as COMPLEX if "
    "the order's price_usd is above $500, the category is final_sale, the customer's message "
    "disputes a policy classification or claims an exception, or the case otherwise sounds "
    "ambiguous. Otherwise classify as ROUTINE."
)
TRIAGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"complexity": {"type": "string", "enum": ["ROUTINE", "COMPLEX"]},
                       "reason": {"type": "string", "description": "One short sentence."}},
        "required": ["complexity", "reason"], "additionalProperties": False,
    },
}


def triage(order: dict, customer_message: str):
    prompt = f"Order: price_usd={order['price_usd']}, category={order['category']}.\nCustomer message: {customer_message}"
    resp = client.messages.create(model=MODEL_HAIKU, max_tokens=150, system=TRIAGE_SYSTEM,
                                  messages=[{"role": "user", "content": prompt}],
                                  output_config={"format": TRIAGE_SCHEMA})
    verdict = json.loads(text_of(resp))
    return verdict.get("complexity", "COMPLEX"), resp  # fail safe: unknown -> COMPLEX


rows = []
for t in TASKS:
    order = ORDERS[t["order_id"]]
    verdict, resp = triage(order, t["query"])
    rows.append([t["id"], verdict, f"${calculate_cost(MODEL_HAIKU, resp.usage):.5f}"])
print(tabulate(rows, headers=["Task", "Triage", "Triage cost"], tablefmt="simple"))
print("\nA fraction of a cent buys the routing decision. Savings come from what it routes AWAY from Sonnet/Opus.")

# %% [markdown]
# ## Lever 3 — Output discipline + the effort dial
#
# `decide_efficient` (Part 3) already collapses v0's essay-then-extract into one tool loop
# plus one schema-constrained call. Compare it directly against v0 on the same order.

# %%
efficient_demo = decide_efficient(TASKS[0]["query"], model=MODEL_SONNET, system=CACHED_SYSTEM, effort="low")
v0_row = BASELINE["rows"][0]
print(f"Efficient single-pass: {efficient_demo['elapsed']:.1f}s · {len(efficient_demo['calls'])} calls · "
      f"${sum(calculate_cost(m, u) for m, u in efficient_demo['calls']):.5f}")
print(f"v0 on the same task:   {v0_row['elapsed']:.1f}s · {v0_row['calls']} calls · ${v0_row['cost']:.4f}")

# %% [markdown]
# ## Lever 4 — Streaming: TTFT is the UX number

# %%
_stream_messages, _ = run_tool_loop(TASKS[1]["query"], MODEL_SONNET, CACHED_SYSTEM)
# The loop always ends on an assistant turn; append one user turn before the next call,
# same fix as final_decision_call above — see its docstring for why.
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
print(f"Streamed final decision call:  TTFT {_ttft*1000:.0f}ms · TTC {_total*1000:.0f}ms")
print(f"v0's TTC on its post-loop call *was* its TTFT — nothing streamed, no partial output until the whole thing lands.")

print("\nLever 5 (parallelism) is demonstrated directly in Part 7's sprint below.")

# %% [markdown]
# # Part 7 · The optimized v1, then the business case
#
# `CONFIG` is the one dial left in this notebook — turn it to explore other lever
# combinations. Everything else in this build is a complete, working reference rather than
# a stubbed build-along: this is a solo capstone, not a live cohort session, so there's no
# one to fill blanks in with.

# %%
CONFIG = {
    "triage_routing": True,
    "routine_model": MODEL_HAIKU,
    "complex_model": MODEL_SONNET,
    "cache_playbook": True,
    "effort_for_complex": "low",
    "parallel_workers": 4,
}


def everstock_v1(task: dict, config=None) -> dict:
    cfg = config or CONFIG
    order = ORDERS[task["order_id"]]
    model = cfg["routine_model"]
    calls = []
    if cfg["triage_routing"]:
        verdict, tri_resp = triage(order, task["query"])
        calls.append((MODEL_HAIKU, tri_resp.usage))
        model = cfg["complex_model"] if verdict == "COMPLEX" else cfg["routine_model"]
    system = CACHED_SYSTEM if cfg["cache_playbook"] else PLAYBOOK
    effort = cfg.get("effort_for_complex") if model != cfg["routine_model"] else None
    out = decide_efficient(task["query"], model, system, effort=effort)
    out["calls"] = calls + out["calls"]
    return out


def engagement_score(report: dict, baseline: dict) -> int:
    if report["accuracy"] < baseline["accuracy"]:
        return 0
    return round(50 * baseline["cost_per_decision"] / max(report["cost_per_decision"], 1e-9)
                 + 50 * baseline["p50_s"] / max(report["p50_s"], 1e-9))


def run_scorecard(config, label, tasks=TASKS):
    report = run_batch(lambda t: everstock_v1(t, config), tasks, workers=config.get("parallel_workers", 0))
    print_report(report, label)
    score = engagement_score(report, BASELINE)
    gate = "" if report["accuracy"] >= BASELINE["accuracy"] else "  ⛔ accuracy gate failed — score zeroed"
    print(f"\n  ENGAGEMENT SCORE: {score}  (v0 baseline = 100){gate}")
    return report


print("Sequential v1:")
v1_sequential = run_scorecard({**CONFIG, "parallel_workers": 0}, "Everstock v1 — sequential")

print("\nParallel v1 (same CONFIG, 4 workers — this is Lever 5):")
OPTIMIZED = run_scorecard(CONFIG, "Everstock v1 — parallel x4")

# %% [markdown]
# ### Holdout — did this generalize, or did it overfit two categories of tuning?

# %%
RUN_HOLDOUT = True
if RUN_HOLDOUT:
    run_scorecard(CONFIG, "Everstock v1 — HOLDOUT", tasks=HOLDOUT_TASKS)
else:
    print("Holdout armed. Set RUN_HOLDOUT = True once CONFIG is final.")

# %% [markdown]
# ### The business case

# %%
ASSUMPTIONS = {
    "daily_return_volume": 5000,
    "manual_review_minutes_per_case": 6,
    "manual_reviewer_hourly_usd": 28,
}


def business_case(baseline: dict, optimized: dict, a=ASSUMPTIONS):
    monthly_volume = a["daily_return_volume"] * 30
    ai_cogs_v0 = baseline["cost_per_decision"] * monthly_volume
    ai_cogs_v1 = optimized["cost_per_decision"] * monthly_volume
    manual_cogs = (a["manual_reviewer_hourly_usd"] / 60 * a["manual_review_minutes_per_case"]) * monthly_volume

    rows = [
        ["Accuracy (audited checks)", f"{baseline['accuracy']*100:.0f}%", f"{optimized['accuracy']*100:.0f}%", "gate held"],
        ["p50 latency / decision", f"{baseline['p50_s']:.1f}s", f"{optimized['p50_s']:.1f}s",
         f"{baseline['p50_s']/max(optimized['p50_s'],1e-9):.1f}x faster"],
        ["Cost / decision", f"${baseline['cost_per_decision']:.4f}", f"${optimized['cost_per_decision']:.4f}",
         f"{baseline['cost_per_decision']/max(optimized['cost_per_decision'],1e-9):.1f}x cheaper"],
        ["Monthly AI COGS (est. volume)", f"${ai_cogs_v0:,.0f}", f"${ai_cogs_v1:,.0f}", f"${ai_cogs_v0-ai_cogs_v1:,.0f} saved"],
        ["vs. fully-manual review baseline", f"${manual_cogs:,.0f}/mo", f"${manual_cogs:,.0f}/mo",
         f"v1 is {manual_cogs/max(ai_cogs_v1,1e-9):.0f}x cheaper than manual"],
    ]
    print("EVERSTOCK RETURNS DECISIONING — before / after")
    print(tabulate(rows, headers=["Metric", "v0 (naive)", "Optimized v1", "Delta"], tablefmt="grid"))
    print("\nAssumptions: " + ", ".join(f"{k}={v}" for k, v in a.items()))


business_case(BASELINE, OPTIMIZED)

# %% [markdown]
# ---
# ## What this demonstrates, end to end
#
# | Skill | Where |
# |---|---|
# | Agentic tool-use loop | Part 3 — `run_tool_loop` |
# | Structured output (`output_config.format`) | Part 3/4 — `DECISION_SCHEMA`, `final_decision_call` |
# | Prompt caching | Part 6 Lever 1 — `CACHED_SYSTEM` |
# | Model routing / portfolio | Part 6 Lever 2 — `triage` + `TRIAGE_SCHEMA` |
# | Effort dial | Part 6 Lever 3 — `effort_for_complex` |
# | Streaming | Part 6 Lever 4 |
# | Parallelism | Part 7 — sequential vs. parallel `v1` |
# | Eval harness from scratch | Part 5 — `TASKS`, `GRADER_REGISTRY`, `run_batch` |
# | LLM-as-judge grading | Part 5 — `grade_no_hallucination` |
# | Accuracy-gated optimization | Part 7 — `engagement_score` |
# | Business translation | Part 7 — `business_case` |
