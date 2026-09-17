# Client Brief — Meridian Support Pilot

*One page. Fill every line. This is what Priya takes to her leadership.*

---

**Client & pilot**
Meridian — an AI agent that triages and resolves customer support tickets (a coordinator routing to billing / technical / account specialists). Live 3 weeks; closing tickets that aren't actually resolved.

**What's actually breaking it**
It's not the model. It's one rule in the coordinator's instructions: *"one ticket, one specialist — never spawn more than one."* Most tickets only raise one issue, so this worked fine in the demo. But real tickets often raise two — like T-4471, where a customer had both a broken SSO login and a billing refund owed. Fixing the login needs one specialist's tools; issuing the refund needs a different specialist's tools. Because the coordinator was told to pick only one, it would fix the issue that specialist could reach, then close the ticket and tell the customer everything was resolved — even though the second issue was never actually touched. That's exactly what happened on the ticket that landed on your desk.

**The fix**
One file, `system-prompt-coordinator.txt`. Two changes to the coordinator's instructions:
1. It can now bring in a second specialist, but only when a ticket has a genuinely separate issue that the first specialist's tools can't reach — not as a default, just when the ticket actually needs it.
2. It's no longer allowed to say a ticket is "resolved" unless every issue was actually fixed. If something can't be fixed, it now says so honestly (escalates) instead of closing the ticket anyway.

No code changes, no model changes — this is entirely a change to how the agent is instructed.

**Proof**
Resolved rate on the ticket you flagged (T-4471), 5 identical runs each time:
- **Before:** 0/5 resolved — $0.13 per run
- **After:** 5/5 resolved — $0.19 per run

Then we tested it on two tickets it had never seen, to make sure the fix wasn't a fluke tuned to one ticket:
- **T-4471, T-4490, T-4503 combined:** 9/9 resolved — $0.18 per run average

The extra ~$0.05–0.06 per ticket is the cost of a second specialist being brought in on the tickets that genuinely need one — not applied across the board. That's the price of the agent stopping short of a false "resolved."

**What it would take**
No re-architecture. The fix already lives in the production prompt file and is validated against held-out tickets it wasn't tuned on. Recommended next step is a short QA pass across a broader sample of real historical tickets (beyond the 3 used here) to confirm the resolved rate holds at scale before removing the human safety net entirely.

**The objection we'll get**
*"Why not just pay for a better model?"* We tested that directly — same ticket, same failure, run on Opus, our largest available model: still 0/5 resolved, at nearly 2x the cost per run ($0.25 vs $0.13). The model was never the bottleneck; the instructions we gave it were. A bigger model would have cost more and failed the same way.


