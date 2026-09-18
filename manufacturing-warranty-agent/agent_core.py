"""Ridgeline Manufacturing Warranty Root-Cause Agent — core library.

Import-safe: defining this module costs nothing except one cheap 1-token
credential-check ping (same pattern as every notebook in this repo). No v0/v1
run, no eval sweep, no context-engineering demo executes on import — those are
narrative/experiment code that lives in Manufacturing_Warranty_Agent.py and
app.py, both of which import from here.
"""

import importlib.util
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional

# ── Dependencies ──────────────────────────────────────────────────────────────

def _ensure_packages(requirements):
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
        f"  (pip said: {tail})\n"
    )


_ensure_packages([("anthropic", "anthropic"), ("tabulate", "tabulate")])

import anthropic
from anthropic.types import TextBlock, ToolUseBlock
from anthropic.types.beta import BetaToolUseBlock
from tabulate import tabulate

# ── Connect to Claude — same shared root .env as every exercise in this repo ──

_ENV_TEMPLATE = (
    "# Anthropic API key — paste after the = (no quotes, no spaces), then save and\n"
    "# re-run. Get one at https://console.anthropic.com/\n"
    "ANTHROPIC_API_KEY=paste-your-key-here\n"
    "\n"
    "# --- Using Amazon Bedrock instead? Comment out the line above and fill these in:\n"
    "# AWS_BEARER_TOKEN_BEDROCK=paste-your-bedrock-api-key-here\n"
    "# AWS_REGION=us-east-1\n"
)


def _resolve_env_file():
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
    print(f"Created {_env_file.name} in {_env_file.parent} — open it, add your key, save, then re-run.")

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
    raise SystemExit(
        f"Add your credentials to {_env_file} — either ANTHROPIC_API_KEY=sk-ant-... "
        "or AWS_BEARER_TOKEN_BEDROCK + AWS_REGION — then re-run."
    )

if PROVIDER == "bedrock" and not _bedrock_region:
    raise SystemExit(f"Found AWS_BEARER_TOKEN_BEDROCK but no AWS_REGION. Add it to {_env_file}.")


def _model(name):
    return f"anthropic.{name}" if PROVIDER == "bedrock" else name


MODEL_HAIKU = _model("claude-haiku-4-5")
MODEL_SONNET = _model("claude-sonnet-5")
MODEL_OPUS = _model("claude-opus-4-8")


def _make_client(timeout, max_retries=2):
    if PROVIDER == "bedrock":
        from anthropic import AnthropicBedrockMantle
        return AnthropicBedrockMantle(aws_region=_bedrock_region, timeout=timeout, max_retries=max_retries)
    return anthropic.Anthropic(api_key=_anthropic_key, timeout=timeout, max_retries=max_retries)


_probe = _make_client(timeout=30.0, max_retries=1)
try:
    _probe.messages.create(model=MODEL_HAIKU, max_tokens=1, messages=[{"role": "user", "content": "ping"}])
except anthropic.NotFoundError:
    raise SystemExit(f"Model '{MODEL_HAIKU}' not found for this key/region.")
except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
    raise SystemExit("Credentials rejected — check your key/token and re-run.")
else:
    if PROVIDER == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = _anthropic_key

client = _make_client(timeout=900.0)
print(f"agent_core ready — SDK {anthropic.__version__} · provider {PROVIDER} · "
      f"portfolio: {MODEL_HAIKU}, {MODEL_SONNET}, {MODEL_OPUS}")

# ── Part 1: Scenario data ──────────────────────────────────────────────────────

TODAY = date(2026, 9, 18)


def _days_between(d1: str, d2: str) -> int:
    y1, m1, dd1 = map(int, d1.split("-"))
    y2, m2, dd2 = map(int, d2.split("-"))
    return (date(y2, m2, dd2) - date(y1, m1, dd1)).days


WARRANTY_TERMS = {
    "standard_duty": {
        "series_name": "Ridgeline SD-Series",
        "warranty_months_from_install": 12,
        "warranty_months_from_ship_cap": 18,
        "wear_part_months_from_install": 6,
        "policy_text": (
            "STANDARD DUTY (SD-SERIES) WARRANTY: Covers material and workmanship defects "
            "(housing, rod, bore, welds) for 12 months from installation, capped at 18 months "
            "from ship date, whichever comes first. Seals, wipers, and o-rings are wear parts "
            "covered for 6 months from installation only. Excluded: improper installation, "
            "operation beyond rated pressure, customer-side fluid contamination, and normal "
            "wear beyond the wear-part window."
        ),
    },
    "heavy_duty": {
        "series_name": "Ridgeline HD-Series",
        "warranty_months_from_install": 24,
        "warranty_months_from_ship_cap": 30,
        "wear_part_months_from_install": 12,
        "policy_text": (
            "HEAVY DUTY (HD-SERIES) WARRANTY: Covers material and workmanship defects, plus "
            "hydraulic seal failure within the first 12 months of installation, for 24 months "
            "from installation, capped at 30 months from ship date. Seals, wipers, and o-rings "
            "are wear parts covered for 12 months from installation. Excluded: contamination-"
            "induced wear from the customer's own hydraulic fluid, misapplication outside rated "
            "pressure or temperature, and force-majeure or impact damage."
        ),
    },
    "severe_service": {
        "series_name": "Ridgeline SS-Series",
        "warranty_months_from_install": 36,
        "warranty_months_from_ship_cap": 42,
        "wear_part_months_from_install": 18,
        "policy_text": (
            "SEVERE SERVICE (SS-SERIES) WARRANTY: Covers material and workmanship defects, "
            "hydraulic seal failure within 18 months of installation, and corrosion-resistant "
            "coating defects, for 36 months from installation, capped at 42 months from ship "
            "date. Seals, wipers, and o-rings are wear parts covered for 18 months from "
            "installation. Excluded: misapplication outside rated specification, contamination, "
            "and unauthorized field modification."
        ),
    },
}

CLAIMS = {
    # -- demo claims (Part 4 live trace, not eval-scored) --
    "CLM-9001": {"claim_id": "CLM-9001", "unit_serial": "SD-70021", "product_line": "standard_duty",
                 "part_number": "SD-CYL-100", "unit_value_usd": 820, "ship_date": "2026-05-01",
                 "install_date": "2026-06-01", "reported_date": "2026-09-10", "operating_hours_at_report": 410,
                 "symptom_description": "Cylinder won't fully retract, seems to bind partway."},
    "CLM-9002": {"claim_id": "CLM-9002", "unit_serial": "SD-61188", "product_line": "standard_duty",
                 "part_number": "SD-CYL-120", "unit_value_usd": 780, "ship_date": "2024-11-01",
                 "install_date": "2024-12-01", "reported_date": "2026-09-12", "operating_hours_at_report": 6200,
                 "symptom_description": "Some seal seepage, unit's been in service a long time."},
    # -- in-sample eval claims --
    "CLM-1001": {"claim_id": "CLM-1001", "unit_serial": "SD-77410", "product_line": "standard_duty",
                 "part_number": "SD-CYL-100", "unit_value_usd": 820, "ship_date": "2026-05-15",
                 "install_date": "2026-06-01", "reported_date": "2026-09-15", "operating_hours_at_report": 520,
                 "symptom_description": "Solenoid valve intermittently fails to actuate; rest of the unit seems fine."},
    "CLM-1002": {"claim_id": "CLM-1002", "unit_serial": "HD-92034", "product_line": "heavy_duty",
                 "part_number": "HD-CYL-450", "unit_value_usd": 2650, "ship_date": "2026-03-20",
                 "install_date": "2026-04-10", "reported_date": "2026-09-16", "operating_hours_at_report": 385,
                 "symptom_description": ("Noticing a slow buildup of oily residue near where the rod exits the "
                                         "cylinder after roughly 380 hours of operation. Nothing sudden, just a "
                                         "steady damp patch that keeps growing.")},
    "CLM-1003": {"claim_id": "CLM-1003", "unit_serial": "HD-55187", "product_line": "heavy_duty",
                 "part_number": "HD-CYL-450", "unit_value_usd": 2650, "ship_date": "2026-08-20",
                 "install_date": "2026-09-01", "reported_date": "2026-09-08", "operating_hours_at_report": 40,
                 "symptom_description": ("Hydraulic fluid on the floor near the rod-end fitting. Started within "
                                         "the first week after we installed it — installer mentioned the fitting "
                                         "felt very tight going on.")},
    "CLM-1004": {"claim_id": "CLM-1004", "unit_serial": "SD-40217", "product_line": "standard_duty",
                 "part_number": "SD-CYL-140", "unit_value_usd": 810, "ship_date": "2025-05-10",
                 "install_date": "2025-06-18", "reported_date": "2026-09-14", "operating_hours_at_report": 3100,
                 "symptom_description": "Housing has developed a small crack, unit still runs but customer wants it fixed."},
    "CLM-1005": {"claim_id": "CLM-1005", "unit_serial": "HD-38820", "product_line": "heavy_duty",
                 "part_number": "HD-CYL-300", "unit_value_usd": 2400, "ship_date": "2025-06-15",
                 "install_date": "2025-07-18", "reported_date": "2026-09-15", "operating_hours_at_report": 2900,
                 "symptom_description": "Rod seal is weeping, otherwise the unit performs fine."},
    "CLM-1006": {"claim_id": "CLM-1006", "unit_serial": "SS-11045", "product_line": "severe_service",
                 "part_number": "SS-CYL-800", "unit_value_usd": 8200, "ship_date": "2026-04-25",
                 "install_date": "2026-05-20", "reported_date": "2026-09-16", "operating_hours_at_report": 610,
                 "symptom_description": ("Performance seems a little off lately — hard to pin down exactly what's "
                                         "wrong. Could be normal wear, could be something starting to fail.")},
    "CLM-1007": {"claim_id": "CLM-1007", "unit_serial": "HD-27703", "product_line": "heavy_duty",
                 "part_number": "HD-CYL-300", "unit_value_usd": 2400, "ship_date": "2026-02-10",
                 "install_date": "2026-03-05", "reported_date": "2026-09-14", "operating_hours_at_report": 1500,
                 "symptom_description": ("Unit failed early. When we opened it up the fluid was visibly dirty with "
                                         "particulates — looks like it came from the customer's own reservoir.")},
    "CLM-1008": {"claim_id": "CLM-1008", "unit_serial": "SS-30091", "product_line": "severe_service",
                 "part_number": "SS-CYL-820", "unit_value_usd": 9100, "ship_date": "2026-01-15",
                 "install_date": "2026-02-10", "reported_date": "2026-09-13", "operating_hours_at_report": 2200,
                 "symptom_description": ("Premature failure. Pressure logs from the customer's PLC show sustained "
                                         "operation above this model's rated pressure spec.")},
    "CLM-1009": {"claim_id": "CLM-1009", "unit_serial": "SD-88802", "product_line": "standard_duty",
                 "part_number": "SD-CYL-120", "unit_value_usd": 780, "ship_date": "2026-06-01",
                 "install_date": "2026-06-20", "reported_date": "2026-09-16", "operating_hours_at_report": 340,
                 "symptom_description": ("Minor leak. Customer says it's probably the same seal problem everyone's "
                                         "been talking about.")},
    # -- holdout claims --
    "CLM-1101": {"claim_id": "CLM-1101", "unit_serial": "SD-91220", "product_line": "standard_duty",
                 "part_number": "SD-CYL-100", "unit_value_usd": 820, "ship_date": "2026-06-10",
                 "install_date": "2026-07-01", "reported_date": "2026-09-17", "operating_hours_at_report": 280,
                 "symptom_description": "Sensor cable connector looks damaged, unit throws an intermittent fault."},
    "CLM-1102": {"claim_id": "CLM-1102", "unit_serial": "HD-60455", "product_line": "heavy_duty",
                 "part_number": "HD-CYL-450", "unit_value_usd": 2650, "ship_date": "2026-05-05",
                 "install_date": "2026-05-28", "reported_date": "2026-09-17", "operating_hours_at_report": 355,
                 "symptom_description": ("Slow, steady oil weep right at the rod seal, been building for a while — "
                                         "roughly 350 hours of runtime so far, nothing sudden about it.")},
}


def _pattern_ticket(ticket_id, serial, phrase, reported_date, hours):
    return {"ticket_id": ticket_id, "product_line": "heavy_duty", "part_number": "HD-CYL-450",
            "unit_serial": serial, "lot_code": "LOT-4471", "reported_date": reported_date,
            "operating_hours_at_failure": hours, "technician_free_text": phrase,
            "root_cause_code": "SEAL-EXTRUSION-4471",
            "root_cause_description": ("Rod seal extrusion caused by an out-of-spec nitrile durometer batch "
                                       "(Lot 4471); the seal gradually loses sealing force, producing a slow "
                                       "external leak, typically between 300-500 operating hours."),
            "resolution_action": "Seal kit replaced with corrected-durometer stock under warranty; lot flagged for engineering review."}


def _red_herring_ticket(ticket_id, serial, phrase, reported_date, hours, lot):
    return {"ticket_id": ticket_id, "product_line": "heavy_duty", "part_number": "HD-CYL-450",
            "unit_serial": serial, "lot_code": lot, "reported_date": reported_date,
            "operating_hours_at_failure": hours, "technician_free_text": phrase,
            "root_cause_code": "INSTALL-THREAD-DAMAGE",
            "root_cause_description": ("Installer over-torqued the rod-end fitting during installation, "
                                       "deforming the threads and creating a leak path — not a seal or material "
                                       "defect. Onset is immediate (within days of install), unlike the gradual "
                                       "300-500-hour seal-extrusion pattern seen in Lot 4471 units."),
            "resolution_action": "Fitting replaced, installer retrained on torque spec; billable — installation error, not covered."}


_PATTERN_TICKETS = [
    _pattern_ticket("TCK-1018", "HD-71203",
                    "Noticed a slow weep of hydraulic oil right where the rod exits the cylinder after about "
                    "350 operating hours. Nothing dramatic, just a steady damp spot building up.",
                    "2026-07-02", 350),
    _pattern_ticket("TCK-1027", "HD-71890",
                    "Actuator is gradually losing pressure — found seepage right at the rod seal after roughly "
                    "400 hours in service.", "2026-07-18", 400),
    _pattern_ticket("TCK-1033", "HD-72410",
                    "There's damp residue collecting near the rod boot, showed up after about 380 hours of use. "
                    "Slow but steady, getting a little worse each week.", "2026-08-01", 380),
    _pattern_ticket("TCK-1038", "HD-73015",
                    "Gradual fluid loss on this unit, looks like extrusion damage on the nitrile rod seal — "
                    "started around 320 operating hours.", "2026-08-14", 320),
    _pattern_ticket("TCK-1046", "HD-73802",
                    "Small but steady wetness building up near where the rod comes out of the cylinder, been "
                    "getting worse since about 300 hours of runtime.", "2026-08-28", 300),
]

_RED_HERRING_TICKETS = [
    _red_herring_ticket("TCK-1004", "HD-69910",
                        "Leak right at the rod-end fitting, started within the first few days after we "
                        "installed it. Fitting looked over-tightened when we inspected it.",
                        "2026-06-05", 25, "LOT-3390"),
    _red_herring_ticket("TCK-1012", "HD-70340",
                        "Hydraulic fluid on the floor near the rod end, happened almost immediately after "
                        "installation. Turned out the fitting threads were damaged.",
                        "2026-06-20", 18, None),
]

_FILLER_ARCHETYPES = [
    {"code": "SOLENOID-FAULT", "desc": "Solenoid valve failed to actuate reliably; not a cylinder defect.",
     "resolution": "Replaced solenoid coil and retested; unit operating normally.",
     "phrases": ["valve doesn't always kick on when it should", "solenoid seems sticky, sometimes doesn't fire",
                 "intermittent actuation failure on the control valve"]},
    {"code": "ROD-BEND-IMPACT", "desc": "Piston rod bent from an impact event during operation, not a material defect.",
     "resolution": "Replaced rod assembly; advised customer on machine guarding.",
     "phrases": ["rod looks bent, might have gotten hit by something", "rod isn't straight anymore after a forklift incident",
                 "visible bend in the rod, unit still moves but binds"]},
    {"code": "CORROSION-PITTING", "desc": "Surface corrosion and pitting on the rod from a corrosive environment outside spec.",
     "resolution": "Rod replacement; recommended coating upgrade.",
     "phrases": ["rust spots forming on the rod surface", "pitting visible along the exposed rod",
                 "corrosion damage near the rod seal area"]},
    {"code": "SEAL-WEAROUT-EOL", "desc": "Ordinary end-of-life seal wear-out, well past the wear-part coverage window.",
     "resolution": "Billable seal kit replacement.",
     "phrases": ["seals just look worn out after years of use", "some seepage, unit has a lot of hours on it",
                 "expected wear after this many duty cycles"]},
    {"code": "COATING-DEFECT", "desc": "Corrosion-resistant coating flaking prematurely — a coating application defect.",
     "resolution": "Unit recoated under warranty.",
     "phrases": ["coating is peeling off in patches", "finish looks like it's flaking near the base",
                 "protective coating failed early"]},
    {"code": "HOSE-FITTING-FAILURE", "desc": "Hydraulic hose fitting failure unrelated to the cylinder itself.",
     "resolution": "Fitting replaced; cylinder inspected and found sound.",
     "phrases": ["leak is actually coming from the hose connection, not the cylinder", "fitting looks like it's weeping",
                 "hose end fitting failed"]},
    {"code": "SENSOR-FAULT", "desc": "Position sensor malfunction, intermittent signal dropout.",
     "resolution": "Sensor replaced under warranty.",
     "phrases": ["position feedback keeps glitching", "sensor reading jumps around randomly",
                 "intermittent signal loss from the position sensor"]},
    {"code": "SHIPPING-DAMAGE", "desc": "Cosmetic and functional damage consistent with mishandling in transit.",
     "resolution": "Replacement unit shipped; freight claim filed.",
     "phrases": ["arrived with a dent, looks like it was dropped", "packaging was crushed, unit doesn't move smoothly",
                 "damage consistent with rough handling in shipping"]},
    {"code": "CONTAMINATION-WEAR", "desc": "Accelerated internal wear from particulate contamination in the customer's own hydraulic fluid.",
     "resolution": "Billable rebuild; recommended fluid filtration audit.",
     "phrases": ["fluid looked dirty when we opened it up", "internal wear pattern suggests contaminated fluid",
                 "particulates found in the hydraulic circuit"]},
    {"code": "MISAPPLICATION-OVERPRESSURE", "desc": "Operating pressure logs show sustained operation above the rated spec.",
     "resolution": "Billable repair; recommended pressure relief valve check.",
     "phrases": ["logs show pressure spikes above rated max", "customer admits they may be running it harder than spec",
                 "operating pressure looks too high for this model"]},
    {"code": "WELD-DEFECT", "desc": "Manufacturing weld defect on the housing — a genuine one-off material defect.",
     "resolution": "Unit replaced under warranty.",
     "phrases": ["there's a crack near one of the welds", "weld seam looks like it's separating",
                 "housing weld failed"]},
    {"code": "BORE-SCORING", "desc": "Scoring inside the cylinder bore from a one-off foreign-object ingress event.",
     "resolution": "Bore honed and reseal under warranty; isolated debris-ingress incident, not a pattern.",
     "phrases": ["scratches visible inside when disassembled", "bore has scoring marks",
                 "something got inside and scored the bore"]},
]

_FILLER_PARTS = ["SD-CYL-100", "SD-CYL-120", "SD-CYL-140", "HD-CYL-300", "HD-CYL-500", "SS-CYL-800", "SS-CYL-820"]
_FILLER_PRODUCT_LINE = {"SD-CYL-100": "standard_duty", "SD-CYL-120": "standard_duty", "SD-CYL-140": "standard_duty",
                        "HD-CYL-300": "heavy_duty", "HD-CYL-500": "heavy_duty",
                        "SS-CYL-800": "severe_service", "SS-CYL-820": "severe_service"}


def _build_filler_tickets(n=43):
    """Deterministic (no randomness — reproducible search results every run), cycling
    through a small set of failure archetypes and parts to fill out the corpus with
    plausible noise that isn't part of the graded pattern."""
    tickets = []
    for i in range(n):
        arch = _FILLER_ARCHETYPES[i % len(_FILLER_ARCHETYPES)]
        phrase = arch["phrases"][i % len(arch["phrases"])]
        part = _FILLER_PARTS[i % len(_FILLER_PARTS)]
        month = 3 + (i % 6)
        day = 1 + (i % 27)
        tickets.append({
            "ticket_id": f"TCK-{2000 + i}", "product_line": _FILLER_PRODUCT_LINE[part], "part_number": part,
            "unit_serial": f"{part[:2]}-{50000 + i * 37}", "lot_code": None,
            "reported_date": f"2026-{month:02d}-{day:02d}", "operating_hours_at_failure": 200 + (i * 61) % 4000,
            "technician_free_text": phrase, "root_cause_code": arch["code"],
            "root_cause_description": arch["desc"], "resolution_action": arch["resolution"],
        })
    return tickets


# Deliberate ordering: the 2 red herrings sit early (positions ~4/12), the 5 real
# pattern tickets sit late (~18/27/33/38/46) — this is what makes v0's fixed early
# slice (Part 5) miss the pattern and catch a red herring by construction, not luck.
_filler = _build_filler_tickets(43)
_ticket_list = (
    _filler[:3] + [_RED_HERRING_TICKETS[0]] + _filler[3:10] + [_RED_HERRING_TICKETS[1]]
    + _filler[10:16] + [_PATTERN_TICKETS[0]] + _filler[16:24] + [_PATTERN_TICKETS[1]]
    + _filler[24:29] + [_PATTERN_TICKETS[2]] + _filler[29:34] + [_PATTERN_TICKETS[3]]
    + _filler[34:41] + [_PATTERN_TICKETS[4]] + _filler[41:]
)
HISTORICAL_TICKETS = {t["ticket_id"]: t for t in _ticket_list}
PATTERN_TICKET_IDS = {t["ticket_id"] for t in _PATTERN_TICKETS}
RED_HERRING_TICKET_IDS = {t["ticket_id"] for t in _RED_HERRING_TICKETS}
V0_STUFFED_SLICE = list(HISTORICAL_TICKETS.values())[:14]  # what a naive v0 "eyeballs" instead of searching

print(f"Loaded {len(WARRANTY_TERMS)} product lines, {len(CLAIMS)} claims, "
      f"{len(HISTORICAL_TICKETS)} historical tickets "
      f"({len(PATTERN_TICKET_IDS)} real pattern, {len(RED_HERRING_TICKET_IDS)} red herring)")

# ── The playbook — cacheable decision framework; numbers live only behind tools ─

PLAYBOOK_CORE = """You are the Warranty Root-Cause Agent for Ridgeline Manufacturing, which makes \
precision hydraulic cylinder actuators. For each claim you determine the root cause, whether it's \
a known recurring failure pattern or a one-off, warranty coverage, and the recommended action — \
justified only by facts you looked up this turn, never from memory or a prior case.

## Decisions (recommended_action)
- REPAIR_UNDER_WARRANTY — covered, no charge to the customer.
- REPAIR_BILLABLE — not covered (out of window, wear-part sub-limit expired, or an exclusion applies).
- REPLACE_UNDER_WARRANTY — covered, unit is replaced rather than repaired.
- ESCALATE_TO_ENGINEERING — you cannot responsibly decide: a suspected or confirmed recurring \
pattern, or a high-value / ambiguous claim.

## Required tool calls, every single time
1. Call get_claim first. Never assume the product line, install date, or symptom details from \
the technician's message alone.
2. Call get_warranty_terms for the claim's product line on every request, even for a line you \
handled a moment ago. Your warranty_citation must be an exact or near-exact quote from that \
tool's output, never from memory.
3. Call search_historical_tickets before concluding there is or isn't a known pattern. Never \
decide pattern_match from your own impression of "this sounds familiar" — the historical corpus \
has 10,000+ tickets in production (this lab uses a representative sample); a pattern can only be \
confirmed by tickets you actually retrieved this turn. Narrow with part_number when you have it.
4. Call get_ticket_detail on any promising search hit before relying on it — a snippet is not \
enough to confirm a match. Two tickets can share a part number and similar-sounding language \
while having completely different root causes; only the full detail (onset timing, lot code, \
root cause) tells you which.

## Escalation and coverage rules
- If get_warranty_terms's response does not define a wear-part sub-limit for the failed \
component, treat the main warranty window as controlling. If it does define one and the \
component is a wear part (seal, wiper, o-ring), the wear-part sub-limit controls even if the \
main warranty is still active — a seal past its own wear-part window is billable even on an \
in-warranty unit.
- If search_historical_tickets and get_ticket_detail together show 2 or more tickets with the \
same root cause as this claim's likely cause, treat it as a recurring pattern (pattern_match \
CONFIRMED_RECURRING or SUSPECTED_EMERGING) and escalate to engineering — do not simply repair it \
as a one-off.
- Do not call something a pattern because it shares a part number or similar-sounding symptom \
language with other tickets. Confirm the actual root cause in each candidate ticket's detail \
matches before citing it — a different lot, a different onset timing (immediate vs. gradual), or \
a different root_cause_code means it is NOT the same pattern, no matter how similar the wording.
- Any exclusion (contamination, misapplication/overpressure, unauthorized modification, improper \
installation) makes a claim billable regardless of how much warranty time remains.
- If a unit's unit_value_usd is above $5,000 and the failure description is ambiguous or \
inconclusive even after your investigation, escalate to a human rather than guessing.

## Evidence discipline
Your final warranty_citation must be traceable to get_warranty_terms's actual returned text. \
Your matched_ticket_ids must be real ticket IDs whose actual detail (not just the search \
snippet) supports the pattern claim — never cite a ticket you didn't call get_ticket_detail on, \
and never cite one whose real root cause doesn't match. A reviewer will spot-check every citation \
against your own transcript.

## Worked examples
Example A: an in-warranty claim with a one-off root cause and no similar tickets found → \
REPAIR_UNDER_WARRANTY, pattern_match NONE.
Example B: a wear-part failure past its own wear-part sub-limit, even though the main warranty \
is still active → REPAIR_BILLABLE, citing the wear-part clause specifically.
Example C: three tickets found and detail-confirmed with the same root cause as this claim → \
ESCALATE_TO_ENGINEERING, pattern_match CONFIRMED_RECURRING, matched_ticket_ids listing those \
three real IDs.
Example D: a ticket search turns up a similar-sounding but detail-confirmed different root cause \
→ pattern_match NONE, and the decision does not cite that ticket as a match.
"""

_PLAYBOOK_NOTES = [
    "Gradual leaks that build slowly over hundreds of operating hours are a different failure "
    "signature than a leak that appears within days of installation — onset timing is often the "
    "fastest way to rule a candidate pattern in or out.",
    "A part number match alone is not a pattern match — most parts have far more one-off tickets "
    "than pattern tickets against them.",
    "Lot codes, when present in a ticket's detail, are strong pattern evidence; their absence "
    "doesn't rule a pattern out, but their presence pointing to a *different* lot than other "
    "candidates strongly rules one in.",
    "High operating-hour, long-service-life seal wear is expected and billable — do not confuse "
    "ordinary end-of-life wear-out with an emerging defect pattern.",
    "A customer's own theory about the cause ('probably the same issue as before') is not "
    "evidence — only your own tool calls this turn are.",
]


def _build_playbook() -> str:
    parts = [PLAYBOOK_CORE, "\n## Investigation notes\n"] + [f"- {n}" for n in _PLAYBOOK_NOTES]
    text = "\n".join(parts)
    annex = "\n\n## Annex: investigation notes (re-issued for length)\n" + "\n".join(f"- {n}" for n in _PLAYBOOK_NOTES)
    while len(text) < 20000:
        text += annex
    return text


PLAYBOOK = _build_playbook()
CACHED_SYSTEM = [{"type": "text", "text": PLAYBOOK, "cache_control": {"type": "ephemeral"}}]
print(f"Playbook built: {len(PLAYBOOK):,} chars (~{len(PLAYBOOK)//4:,} tokens)")

# ── Tools ───────────────────────────────────────────────────────────────────────

_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "it", "was", "were",
             "this", "that", "near", "after", "about", "been", "has", "have", "with", "from", "for",
             "right", "just", "looks", "looked", "like", "some", "much", "getting", "since", "almost",
             "turned", "out", "unit", "when", "opened", "found", "seems", "still", "there", "went"}


def _tokenize(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOPWORDS]


for _t in HISTORICAL_TICKETS.values():
    _t["_tokens"] = set(_tokenize(_t["technician_free_text"] + " " + _t["root_cause_description"]))


def get_claim(claim_id: str) -> dict:
    c = CLAIMS.get(claim_id)
    if c is None:
        return {"error": f"No claim found with id {claim_id}"}
    return {**{k: v for k, v in c.items() if k != "_tokens"},
            "days_since_install": _days_between(c["install_date"], TODAY.isoformat()),
            "days_since_ship": _days_between(c["ship_date"], TODAY.isoformat()),
            "as_of_date": TODAY.isoformat()}


def get_warranty_terms(product_line: str) -> dict:
    t = WARRANTY_TERMS.get(product_line)
    if t is None:
        return {"error": f"Unknown product_line {product_line}"}
    return dict(t)


def search_historical_tickets(query: str, part_number: Optional[str] = None) -> dict:
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return {"error": "query must contain at least one meaningful keyword"}
    pool = HISTORICAL_TICKETS.values()
    if part_number:
        pool = [t for t in pool if t["part_number"] == part_number]
        if not pool:
            return {"results": [], "note": f"No tickets found for part_number {part_number}. Try broadening the search."}
    scored = []
    for t in pool:
        overlap = len(q_tokens & t["_tokens"])
        if overlap == 0:
            continue
        scored.append((overlap / len(q_tokens), t))
    scored.sort(key=lambda x: (-x[0], x[1]["ticket_id"]))
    top = scored[:5]
    return {"results": [
        {"ticket_id": t["ticket_id"], "product_line": t["product_line"], "part_number": t["part_number"],
         "snippet": t["technician_free_text"][:120], "relevance_score": round(score, 2)}
        for score, t in top
    ], "note": "Call get_ticket_detail on a promising hit before relying on it — a snippet is not confirmation."}


def get_ticket_detail(ticket_id: str) -> dict:
    t = HISTORICAL_TICKETS.get(ticket_id)
    if t is None:
        return {"error": f"No ticket found with id {ticket_id}"}
    return {k: v for k, v in t.items() if k != "_tokens"}


TOOL_REGISTRY = {
    "get_claim": get_claim,
    "get_warranty_terms": get_warranty_terms,
    "search_historical_tickets": search_historical_tickets,
    "get_ticket_detail": get_ticket_detail,
}

GET_CLAIM_SPEC = {
    "name": "get_claim",
    "description": (
        "Retrieve the full details of a single warranty claim by its claim ID: unit serial, "
        "product line, part number, unit value, ship/install dates, days since install/ship "
        "(already computed as of today's review date — use directly), operating hours at "
        "report, and the reported symptom. Always call this first — never assume the product "
        "line, install date, or symptom details from the technician's message alone."
    ),
    "input_schema": {"type": "object",
                     "properties": {"claim_id": {"type": "string", "description": "The claim ID, e.g. 'CLM-1002'."}},
                     "required": ["claim_id"], "additionalProperties": False},
}

GET_WARRANTY_TERMS_SPEC = {
    "name": "get_warranty_terms",
    "description": (
        "Look up Ridgeline's official written warranty terms for one product line: the main "
        "warranty window, the ship-date cap, the wear-part sub-limit, and the exact policy "
        "clause text to cite as evidence. Call this for every claim, even if you believe you "
        "already know the line's terms — terms vary by line and your warranty_citation must "
        "quote this tool's output, not a memorized or assumed rule."
    ),
    "input_schema": {"type": "object",
                     "properties": {"product_line": {"type": "string",
                                                      "enum": ["standard_duty", "heavy_duty", "severe_service"],
                                                      "description": "The product line from get_claim's output."}},
                     "required": ["product_line"], "additionalProperties": False},
}

SEARCH_HISTORICAL_TICKETS_SPEC = {
    "name": "search_historical_tickets",
    "description": (
        "Search the historical warranty-ticket corpus (10,000+ tickets in production; this lab "
        "uses a representative sample) for tickets with similar symptoms, using specific symptom "
        "keywords — not the customer's full sentence. Returns only the top 5 most relevant "
        "matches, each with a short snippet, never the full corpus. Narrow with part_number when "
        "known. Never conclude a pattern exists or doesn't exist without calling this tool first "
        "— your own impression of what 'sounds familiar' is not evidence."
    ),
    "input_schema": {"type": "object",
                     "properties": {
                         "query": {"type": "string", "description": "Specific symptom keywords to search for."},
                         "part_number": {"type": "string", "description": "Optional — narrow results to this part number."},
                     },
                     "required": ["query"], "additionalProperties": False},
}

GET_TICKET_DETAIL_SPEC = {
    "name": "get_ticket_detail",
    "description": (
        "Retrieve the full record for one historical ticket by its ticket ID (from a "
        "search_historical_tickets result — never guess an ID): onset timing, lot code, "
        "operating hours at failure, the confirmed root cause code and description, and the "
        "resolution taken. Call this on any promising search hit before citing it as a pattern "
        "match — a snippet alone cannot confirm the root cause actually matches."
    ),
    "input_schema": {"type": "object",
                     "properties": {"ticket_id": {"type": "string", "description": "The ticket ID, e.g. 'TCK-1018'."}},
                     "required": ["ticket_id"], "additionalProperties": False},
}

TOOL_SPECS = [GET_CLAIM_SPEC, GET_WARRANTY_TERMS_SPEC, SEARCH_HISTORICAL_TICKETS_SPEC, GET_TICKET_DETAIL_SPEC]
TOOL_SPECS_V0 = [GET_CLAIM_SPEC, GET_WARRANTY_TERMS_SPEC]  # v0 gets no search/detail — see ridgeline_v0


def execute_tool(name: str, inputs: dict) -> str:
    try:
        return json.dumps(TOOL_REGISTRY[name](**inputs))
    except Exception as e:
        return json.dumps({"error": str(e)})


print(f"Tools ready: {list(TOOL_REGISTRY)}")

# ── Structured decision schema ─────────────────────────────────────────────────

DECISION_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "root_cause_code": {"type": "string"},
            "root_cause_description": {"type": "string"},
            "pattern_match": {"type": "string", "enum": ["NONE", "SUSPECTED_EMERGING", "CONFIRMED_RECURRING"]},
            "matched_ticket_ids": {"type": "array", "items": {"type": "string"},
                                   "description": "Real ticket IDs whose actual detail supports the pattern claim. Empty if pattern_match is NONE."},
            "warranty_covered": {"type": "boolean"},
            "warranty_citation": {"type": "string", "description": "Exact/near-exact quote from get_warranty_terms's output."},
            "recommended_action": {"type": "string",
                                   "enum": ["REPAIR_UNDER_WARRANTY", "REPAIR_BILLABLE", "REPLACE_UNDER_WARRANTY", "ESCALATE_TO_ENGINEERING"]},
            "claim_facts_cited": {"type": "array", "items": {"type": "string"}},
            "reasoning_summary": {"type": "string"},
        },
        "required": ["root_cause_code", "root_cause_description", "pattern_match", "matched_ticket_ids",
                     "warranty_covered", "warranty_citation", "recommended_action", "claim_facts_cited",
                     "reasoning_summary"],
        "additionalProperties": False,
    },
}

TRIAGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"complexity": {"type": "string", "enum": ["ROUTINE", "COMPLEX"]},
                       "reason": {"type": "string", "description": "One short sentence."}},
        "required": ["complexity", "reason"], "additionalProperties": False,
    },
}

JUDGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
                       "reason": {"type": "string", "description": "One sentence, naming the specific evidence."}},
        "required": ["verdict", "reason"], "additionalProperties": False,
    },
}

print("Schemas ready")

# ── The agentic loop ────────────────────────────────────────────────────────────

def text_of(response) -> str:
    return "".join(b.text for b in response.content if isinstance(b, TextBlock))


def extract_json(text: str):
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


def run_tool_loop(query: str, model: str, system, tools=None, max_turns: int = 10):
    """max_turns is a hard safety cap, not a tuning knob — a model that never converges to
    stop_reason != "tool_use" would otherwise loop (and bill) forever."""
    tools = tools if tools is not None else TOOL_SPECS
    messages = [{"role": "user", "content": query}]
    calls = []
    for _ in range(max_turns):
        resp = client.messages.create(model=model, max_tokens=1200, system=system, tools=tools, messages=messages)
        calls.append((model, resp.usage))
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        tool_uses = [b for b in resp.content if isinstance(b, ToolUseBlock)]
        results = [{"type": "tool_result", "tool_use_id": tu.id, "content": execute_tool(tu.name, tu.input)}
                   for tu in tool_uses]
        messages.append({"role": "user", "content": results})
    else:
        raise RuntimeError(f"run_tool_loop exceeded max_turns={max_turns} without converging — "
                           f"last tool calls: {[b.name for b in resp.content if isinstance(b, ToolUseBlock)]}")
    return messages, calls


def collect_tool_calls(messages) -> list:
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


def final_decision_call(messages, model: str, system, tools=None, effort: Optional[str] = None):
    """messages, as returned by run_tool_loop, always ends on an assistant turn (the loop
    appends the assistant's message immediately before checking stop_reason). Current-gen
    models reject a request whose conversation ends on 'assistant' as unsupported prefill —
    so we append one synthetic user turn asking for the decision, locally, without mutating
    the caller's messages list (collect_tool_calls doesn't need to see it)."""
    tools = tools if tools is not None else TOOL_SPECS
    output_config = {"format": DECISION_SCHEMA}
    if effort and model != MODEL_HAIKU:  # effort errors on Haiku 4.5
        output_config["effort"] = effort
    request_messages = messages + [{"role": "user", "content":
        "Based on everything you've found, provide your final decision now, matching the required schema."}]
    resp = client.messages.create(model=model, max_tokens=1500, system=system, tools=tools,
                                  tool_choice={"type": "none"}, messages=request_messages, output_config=output_config)
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("final_decision_call hit max_tokens before completing the schema — raise max_tokens")
    return json.loads(text_of(resp)), resp


def decide_efficient(query: str, model: str, system, effort: Optional[str] = None) -> dict:
    t0 = time.perf_counter()
    messages, calls = run_tool_loop(query, model, system)
    decision, final_resp = final_decision_call(messages, model, system, effort=effort)
    calls.append((model, final_resp.usage))
    return {"decision": decision, "tool_calls": collect_tool_calls(messages), "calls": calls,
            "elapsed": time.perf_counter() - t0, "messages": messages}


def run_tool_loop_ce(query: str, model: str, system, trigger_tokens: int = 3000, keep_tool_uses: int = 2,
                     max_turns: int = 10):
    """Same loop as run_tool_loop, but with context editing enabled once the transcript
    crosses trigger_tokens of input. The client stays fully stateless — the full
    unedited history is sent every turn; the server clears old tool_use/tool_result
    pairs before processing and reports what it cleared in response.context_management.
    Nothing needs to be resynced locally (unlike compaction). max_turns is a hard safety cap:
    clearing too aggressively (e.g. keep_tool_uses=1) can make the model lose track of what
    it already investigated and re-issue the same tool calls indefinitely — a real failure
    mode of this lever, not a hypothetical one, which is exactly why the cap exists."""
    messages = [{"role": "user", "content": query}]
    calls = []
    edit_log = []
    for _ in range(max_turns):
        resp = client.beta.messages.create(
            model=model, max_tokens=1200, system=system, tools=TOOL_SPECS, messages=messages,
            betas=["context-management-2025-06-27"],
            context_management={"edits": [{
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": trigger_tokens},
                "keep": {"type": "tool_uses", "value": keep_tool_uses},
            }]},
        )
        calls.append((model, resp.usage))
        applied = getattr(resp, "context_management", None)
        if applied and getattr(applied, "applied_edits", None):
            edit_log.append({"turn": len(calls), "input_tokens": resp.usage.input_tokens,
                             "applied_edits": [e.model_dump() if hasattr(e, "model_dump") else e
                                              for e in applied.applied_edits]})
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        # client.beta.messages.create returns Beta-typed blocks (BetaToolUseBlock), not the
        # plain anthropic.types.ToolUseBlock the non-beta endpoint returns — checking the
        # wrong type here silently finds zero tool calls and sends an empty tool-results
        # message, which the API rejects.
        tool_uses = [b for b in resp.content if isinstance(b, BetaToolUseBlock)]
        results = [{"type": "tool_result", "tool_use_id": tu.id, "content": execute_tool(tu.name, tu.input)}
                   for tu in tool_uses]
        messages.append({"role": "user", "content": results})
    else:
        raise RuntimeError(f"run_tool_loop_ce exceeded max_turns={max_turns} without converging — "
                           f"aggressive clearing (keep_tool_uses={keep_tool_uses}) may be confusing the model")
    return messages, calls, edit_log


def triage(claim: dict, symptom_message: str):
    prompt = (f"Claim: unit_value_usd={claim['unit_value_usd']}, product_line={claim['product_line']}.\n"
              f"Symptom message: {symptom_message}")
    resp = client.messages.create(
        model=MODEL_HAIKU, max_tokens=150,
        system=("Triage a manufacturing warranty claim. Classify COMPLEX if unit_value_usd is above "
                "$5,000, the symptom sounds ambiguous or disputes a classification, or otherwise sounds "
                "like it could be a recurring pattern (vague leak/seal language). Otherwise ROUTINE."),
        messages=[{"role": "user", "content": prompt}], output_config={"format": TRIAGE_SCHEMA})
    verdict = json.loads(text_of(resp))
    return verdict.get("complexity", "COMPLEX"), resp


print("Agentic loop ready")

# ── v0 (naive baseline) and v1 (optimized, CONFIG-driven) ──────────────────────

def ridgeline_v0(task: dict) -> dict:
    """Deliberately naive: one expensive uncached model, no search tool — instead a
    fixed early slice of the corpus is stuffed into the system prompt (the failure
    mode Part 7's degradation-curve demo measures directly), then an unconstrained
    essay-then-JSON pass instead of a schema."""
    t0 = time.perf_counter()
    slice_text = "\n\n".join(
        f"[{t['ticket_id']} / {t['part_number']}] {t['technician_free_text']} "
        f"(root cause on file: {t['root_cause_code']})"
        for t in V0_STUFFED_SLICE
    )
    system = PLAYBOOK + "\n\n## Reference: recent ticket excerpts (not exhaustive, not searchable)\n" + slice_text
    messages, calls = run_tool_loop(task["query"], MODEL_OPUS, system, tools=TOOL_SPECS_V0)
    messages.append({"role": "user", "content": (
        "Now explain your reasoning step by step in detail, weighing whether this matches any "
        "pattern in the reference excerpts above, and then output a JSON object with keys "
        "root_cause_code, root_cause_description, pattern_match, matched_ticket_ids, "
        "warranty_covered, warranty_citation, recommended_action, claim_facts_cited, "
        "reasoning_summary."
    )})
    resp = client.messages.create(model=MODEL_OPUS, max_tokens=3000, system=system, messages=messages)
    calls.append((MODEL_OPUS, resp.usage))
    messages.append({"role": "assistant", "content": resp.content})
    decision = extract_json(text_of(resp))
    if decision is None and resp.stop_reason == "max_tokens":
        raise RuntimeError("ridgeline_v0 hit max_tokens before completing its essay+JSON pass — raise max_tokens")
    return {"decision": decision, "tool_calls": collect_tool_calls(messages), "calls": calls,
            "elapsed": time.perf_counter() - t0, "messages": messages}


CONFIG = {
    "triage_routing": True,
    "routine_model": MODEL_HAIKU,
    "complex_model": MODEL_SONNET,
    "cache_playbook": True,
    "effort_for_complex": "low",
    "parallel_workers": 4,
    "context_edit_trigger_tokens": 3000,
}


def ridgeline_v1(task: dict, config=None) -> dict:
    cfg = config or CONFIG
    claim = CLAIMS[task["claim_id"]]
    model = cfg["routine_model"]
    calls = []
    if cfg["triage_routing"]:
        verdict, tri_resp = triage(claim, task["query"])
        calls.append((MODEL_HAIKU, tri_resp.usage))
        model = cfg["complex_model"] if verdict == "COMPLEX" else cfg["routine_model"]
    system = CACHED_SYSTEM if cfg["cache_playbook"] else PLAYBOOK
    effort = cfg.get("effort_for_complex") if model != cfg["routine_model"] else None
    out = decide_efficient(task["query"], model, system, effort=effort)
    out["calls"] = calls + out["calls"]
    return out


print("v0/v1 pipelines ready")

# ── Cost accounting ──────────────────────────────────────────────────────────────

PRICING = {
    MODEL_HAIKU: {"input": 1.00, "output": 5.00},
    MODEL_SONNET: {"input": 2.00, "output": 10.00},  # current verified rate — see plan note vs. build #1
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


# ── Eval harness ────────────────────────────────────────────────────────────────

def grade_decision_match(result, check, context=None):
    got = (result["decision"] or {}).get("recommended_action")
    ok = str(got or "").upper() == str(check).upper()
    return {"score": 1.0 if ok else 0.0, "reason": f"recommended_action={got!r} expected={check!r}"}


def grade_pattern_match(result, check, context=None):
    got = (result["decision"] or {}).get("pattern_match")
    ok = str(got or "").upper() == str(check).upper()
    return {"score": 1.0 if ok else 0.0, "reason": f"pattern_match={got!r} expected={check!r}"}


def grade_matched_tickets_overlap(result, check, context=None):
    got = set((result["decision"] or {}).get("matched_ticket_ids") or [])
    if "expect_any_of" in check:
        expected = set(check["expect_any_of"])
        ok = bool(got & expected)
        return {"score": 1.0 if ok else 0.0, "reason": f"matched={sorted(got)}, expected any of {sorted(expected)}"}
    if "expect_none_of" in check:
        forbidden = set(check["expect_none_of"])
        ok = not (got & forbidden)
        return {"score": 1.0 if ok else 0.0, "reason": f"matched={sorted(got)}, must not include {sorted(forbidden)}"}
    return {"score": 0.0, "reason": f"malformed check: {check}"}


def grade_tool_called(result, check, context=None):
    tool_name = check["tool_name"]
    called = any(c["name"] == tool_name for c in result["tool_calls"])
    actual = [c["name"] for c in result["tool_calls"]]
    return {"score": 1.0 if called else 0.0, "reason": f"{tool_name} called={called}; actual={actual}"}


def grade_no_false_pattern_claim(result, check, context=None):
    decision = result["decision"] or {}
    evidence = "\n".join(f"- {c['name']}({c['arguments']}) -> {c.get('result', '')}" for c in result["tool_calls"])
    judge_prompt = (
        "You are auditing a manufacturing warranty agent for false pattern claims. Below are the "
        "ACTUAL tool outputs it saw this turn (ground truth), followed by its final decision.\n\n"
        f"TOOL OUTPUTS:\n{evidence}\n\n"
        f"FINAL DECISION:\npattern_match: {decision.get('pattern_match')}\n"
        f"matched_ticket_ids: {decision.get('matched_ticket_ids')}\n"
        f"root_cause_code: {decision.get('root_cause_code')}\n"
        f"reasoning_summary: {decision.get('reasoning_summary')}\n\n"
        f"Criterion to check: {check}\n\n"
        "Does the final decision satisfy this criterion? Judge only against the tool outputs "
        "above — do not use outside knowledge of what a 'plausible' root cause might be."
    )
    resp = client.messages.create(model=MODEL_HAIKU, max_tokens=300,
                                  messages=[{"role": "user", "content": judge_prompt}],
                                  output_config={"format": JUDGE_SCHEMA})
    data = json.loads(text_of(resp))
    return {"score": 1.0 if data["verdict"] == "PASS" else 0.0, "reason": data["reason"]}


GRADER_REGISTRY = {
    "decision_match": grade_decision_match,
    "pattern_match": grade_pattern_match,
    "matched_tickets_overlap": grade_matched_tickets_overlap,
    "tool_called": grade_tool_called,
    "no_false_pattern_claim": grade_no_false_pattern_claim,
}

TASKS = [
    {"id": "clean_routine_no_pattern", "claim_id": "CLM-1001",
     "query": "Claim CLM-1001: solenoid valve intermittently fails to actuate; rest of the unit seems fine.",
     "graders": [{"type": "decision_match", "checks": ["REPAIR_UNDER_WARRANTY"]},
                 {"type": "pattern_match", "checks": ["NONE"]}]},
    {"id": "needle_in_haystack_pattern_found", "claim_id": "CLM-1002",
     "query": ("Claim CLM-1002: noticing a slow buildup of oily residue near where the rod exits the "
              "cylinder after roughly 380 hours of operation. Nothing sudden, just a steady damp patch."),
     "graders": [{"type": "decision_match", "checks": ["ESCALATE_TO_ENGINEERING"]},
                 {"type": "matched_tickets_overlap", "checks": [{"expect_any_of": list(PATTERN_TICKET_IDS)}]},
                 {"type": "tool_called", "checks": [{"tool_name": "search_historical_tickets"}]}]},
    {"id": "false_positive_trap", "claim_id": "CLM-1003",
     "query": ("Claim CLM-1003: hydraulic fluid on the floor near the rod-end fitting. Started within "
              "the first week after installation — installer mentioned the fitting felt very tight."),
     "graders": [{"type": "matched_tickets_overlap", "checks": [{"expect_none_of": list(PATTERN_TICKET_IDS)}]},
                 {"type": "no_false_pattern_claim", "checks": [
                     "The decision does not claim this claim matches the Lot 4471 gradual seal-extrusion "
                     "pattern, since the actual tool evidence (if investigated) shows immediate onset "
                     "consistent with installation-related thread damage, not that pattern's root cause."]}]},
    {"id": "out_of_warranty_clean_denial", "claim_id": "CLM-1004",
     "query": "Claim CLM-1004: housing has developed a small crack, unit still runs but customer wants it fixed.",
     "graders": [{"type": "decision_match", "checks": ["REPAIR_BILLABLE"]}]},
    {"id": "billable_vs_warranty_edge_case", "claim_id": "CLM-1005",
     "query": "Claim CLM-1005: rod seal is weeping, otherwise the unit performs fine.",
     "graders": [{"type": "decision_match", "checks": ["REPAIR_BILLABLE"]}]},
    {"id": "high_value_ambiguous_escalation", "claim_id": "CLM-1006",
     "query": ("Claim CLM-1006: performance seems a little off lately, hard to pin down exactly what's "
              "wrong. Could be normal wear, could be something starting to fail."),
     "graders": [{"type": "decision_match", "checks": ["ESCALATE_TO_ENGINEERING"]}]},
    {"id": "contamination_exclusion", "claim_id": "CLM-1007",
     "query": ("Claim CLM-1007: unit failed early. Fluid was visibly dirty with particulates when we "
              "opened it up — looks like it came from the customer's own reservoir."),
     "graders": [{"type": "decision_match", "checks": ["REPAIR_BILLABLE"]}]},
    {"id": "misapplication_exclusion_severe_service", "claim_id": "CLM-1008",
     "query": ("Claim CLM-1008: premature failure. Customer's PLC pressure logs show sustained operation "
              "above this model's rated pressure spec."),
     "graders": [{"type": "decision_match", "checks": ["REPAIR_BILLABLE"]}]},
    {"id": "tool_skipped_check", "claim_id": "CLM-1009",
     "query": "Claim CLM-1009: minor leak. Customer says it's probably the same seal problem everyone's been talking about.",
     "graders": [{"type": "tool_called", "checks": [{"tool_name": "search_historical_tickets"}]}]},
]

HOLDOUT_TASKS = [
    {"id": "holdout_clean_approve", "claim_id": "CLM-1101",
     "query": "Claim CLM-1101: sensor cable connector looks damaged, unit throws an intermittent fault.",
     "graders": [{"type": "decision_match", "checks": ["REPAIR_UNDER_WARRANTY"]}]},
    {"id": "holdout_pattern_variant", "claim_id": "CLM-1102",
     "query": ("Claim CLM-1102: slow, steady oil weep right at the rod seal, been building for a while — "
              "roughly 350 hours of runtime so far, nothing sudden about it."),
     "graders": [{"type": "decision_match", "checks": ["ESCALATE_TO_ENGINEERING"]},
                 {"type": "matched_tickets_overlap", "checks": [{"expect_any_of": list(PATTERN_TICKET_IDS)}]}]},
]

print(f"{len(TASKS)} in-sample tasks, {len(HOLDOUT_TASKS)} holdout tasks, graders: {list(GRADER_REGISTRY)}")


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
            rows.append(run_one(head))
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


# ── Pre-flight consistency check (no API calls) ────────────────────────────────

def _verify_corpus():
    errors = []
    if len(PATTERN_TICKET_IDS) != 5:
        errors.append(f"expected 5 pattern tickets, got {len(PATTERN_TICKET_IDS)}")
    if PATTERN_TICKET_IDS & RED_HERRING_TICKET_IDS:
        errors.append("pattern and red-herring ticket sets overlap")
    for label, tasks in [("TASKS", TASKS), ("HOLDOUT_TASKS", HOLDOUT_TASKS)]:
        seen = set()
        for t in tasks:
            if t["id"] in seen:
                errors.append(f"{label}: duplicate id {t['id']}")
            seen.add(t["id"])
            if t["claim_id"] not in CLAIMS:
                errors.append(f"{label}/{t['id']}: claim_id {t['claim_id']} not in CLAIMS")
                continue
            claim = CLAIMS[t["claim_id"]]
            if claim["product_line"] not in WARRANTY_TERMS:
                errors.append(f"{label}/{t['id']}: product_line {claim['product_line']} not in WARRANTY_TERMS")
            if t["claim_id"] not in t["query"]:
                errors.append(f"{label}/{t['id']}: claim_id not mentioned in query text")
            for grader in t["graders"]:
                for check in grader["checks"]:
                    if isinstance(check, dict):
                        for ids in (check.get("expect_any_of", []), check.get("expect_none_of", [])):
                            for tid in ids:
                                if tid not in HISTORICAL_TICKETS:
                                    errors.append(f"{label}/{t['id']}: referenced ticket {tid} not in HISTORICAL_TICKETS")
    if errors:
        raise AssertionError("Corpus/eval consistency check failed:\n" + "\n".join(f" - {e}" for e in errors))
    print(f"✓ Pre-flight check passed — {len(HISTORICAL_TICKETS)} tickets, {len(CLAIMS)} claims, "
          f"{len(TASKS)} tasks, {len(HOLDOUT_TASKS)} holdout, pattern set size {len(PATTERN_TICKET_IDS)}, "
          f"red-herring set size {len(RED_HERRING_TICKET_IDS)} — no dangling references")


_verify_corpus()
