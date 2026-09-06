# HMIS — Supply Chain: LPO, Receiving & Inter-Facility Transfer

## How to use this document

Same model as every prior session: paste the "Mission briefing" as your first message to Claude Code, work phases in order, keep the Process Integrity rules in force. This is a bigger build than recent sessions — it introduces a foundational model (`Facility`) that later phases depend on, so don't skip ahead.

---

## Mission briefing (paste this to Claude Code)

The current supply chain in this HMIS is two disconnected, weak systems: a pharmacy-only Supplier/PurchaseOrder cycle, and a separate internal stores requisition/issue flow for non-pharm items. Neither has real segregation of duties, neither captures real receiving data (both fabricate expiry dates instead of recording what's actually delivered), there's no stock ledger, no budget control, and no concept of more than one facility — so no inter-facility transfer is possible even in principle. This session rebuilds this properly: a unified LPO-to-receiving chassis covering both drugs and non-pharm items, a real stock ledger, and inter-facility transfer built on a new `Facility` model.

### Process Integrity — carried forward, unchanged

- **P.1 Hard stops are hard stops.** `DECISION NEEDED` items get written to `DECISIONS_PENDING.md` and nothing downstream gets built until answered. This has held for the last several sessions — keep it that way, especially here, since procurement controls are exactly the kind of thing where guessing wrong has real institutional consequences.
- **P.2 Evidence over claims.** Real query output, real test names, real before/after — not summary sentences.
- **P.3 No self-certification.** Building segregation-of-duties controls into the code doesn't mean this facility is now PPADA-compliant — that's an institutional/legal determination, not a code state.
- **P.4 Branch and PR discipline.** One phase, one branch, one PR, left open. This has been inconsistent — three violations across the last several sessions against two clean passes. Hold it this time.

---

## PHASE A — Fix the two confirmed active defects first

These are real bugs in the existing code, not new features. Fix before building anything on top of them.

### A.1 Stop fabricating expiry dates
Two places currently invent expiry dates instead of recording real ones:
- `departments/pharmacy/po_routes.py`'s `receive_po_shipment()`: hardcodes `expiry_date = now + 365 days` for every item in every shipment.
- `departments/stores/routes.py`'s `issue_request()`: hardcodes `expiry_date = today + 2 years` when creating a batch.

Both need to become real input fields captured at the point of receiving — batch number and expiry date must come from what's actually printed on the delivered stock, entered by whoever is receiving it, not assumed. This is the single highest-priority fix in this document: it currently feeds FEFO dispensing logic with fabricated dates.
- **Done when**: receiving a shipment (via either path) requires an actual expiry date input before it can be completed, and a test proves a batch can no longer be created with a system-computed placeholder date.

### A.2 Remove leftover debug output and generic exception swallowing
`departments/stores/routes.py` has raw `print()` debug statements throughout and broad `except Exception as e: flash(...); print(...)` blocks that swallow real errors. Replace prints with proper logging (matching the pattern used elsewhere in the app), and narrow the exception handling to what's actually expected (database errors vs. genuinely unexpected ones), consistent with the fix already applied to `app.py`'s exception handling earlier in this project.
- **Done when**: no bare `print()` remains in this file, and a deliberately-triggered unexpected error still surfaces clearly in logs rather than being silently flashed as a generic message.

---

## PHASE B — Foundational: the `Facility` model

Nothing about inter-facility transfer can be built without this existing first.

### B.1 Facility model
Add a `Facility` model: name, facility code (Kenya's KMHFL facility code field, even if not yet integrated with the actual KMHFL API — just the field), facility type (hospital/health center/dispensary — whatever granularity makes sense), address, is_active. Add a `home_facility_id` concept — a config value or a row marked `is_self=True` representing *this* installation, since transfers need to distinguish "us" from "them."
- **Done when**: a facility record can be created, and the current installation's own facility record is seeded/identifiable, tested.

### B.2 — DECISION NEEDED: is this a single-facility or multi-facility deployment?
Before building transfer logic: is this HMIS meant to run as one instance serving one hospital that occasionally ships to/receives from *other institutions running their own separate systems* (in which case a transfer is really "outbound to an external party" and "inbound from an external party," recorded one-sided), or is it meant to eventually run as one shared system across multiple facilities in the same network (in which case a transfer could someday be a single transaction visible to both sides in the same database)? This changes the `Facility` model's shape and whether "the other side" needs its own login/confirmation in this system or is just a reference record. Write this to `DECISIONS_PENDING.md` and stop before Phase E — Phases C and D don't depend on the answer and can proceed.

---

## PHASE C — Unified LPO-to-receiving chassis

Replace both existing disconnected systems with one that covers drugs and non-pharm items alike.

### C.1 Unified models
Extend `Supplier`/`PurchaseOrder`/`PurchaseOrderItem` (already exist, pharmacy-scoped) to reference a generic "stock item" rather than only `Drug` — either a polymorphic reference (item type + item id, covering both `Drug` and `NonPharmItem`) or a shared parent table, whichever fits this codebase's existing patterns better. Don't build a third parallel PO system for non-pharm items — unify onto one.
- **Done when**: a purchase order can contain both drug and non-pharm line items in the same order, tested.

### C.2 Segregation of duties
Currently the same role can create, approve, and receive the same PO. Enforce distinct steps:
- **Originate** (any authorized store/pharmacy staff): creates a DRAFT PO.
- **Approve** (a distinct role or, at minimum, a distinct user from the originator — enforce "not the same user," not just "has an approval-capable role," since two people with the same role is still two people): moves DRAFT → ORDERED. Record who approved and when.
- **Receive** (can be the same person as originator, but log a flag if the receiver is also the approver, for later audit review rather than blocking it outright — full three-way separation may be more strictness than a small facility can staff).
- **Done when**: a test proves the same user cannot both approve and have originated a given PO, and the audit trail on a received PO shows three distinct timestamped actions with three (or at least two) distinct users.

### C.3 Real receiving — partial quantities and real batches
Replace the "always receives 100%" assumption:
- Each line item on receiving records an actual quantity received (which may be less than, equal to, or — flagged for review — more than what was ordered), plus the real batch number and expiry date from Phase A.1.
- A PO with partial receipt across all lines should be in a distinct status (e.g., `PARTIALLY_RECEIVED`) rather than `RECEIVED`, with the outstanding balance still trackable.
- **Done when**: a test receives a PO short of its ordered quantity, confirms the status reflects partial receipt, confirms a second receiving action can complete the remainder, and confirms stock only increases by what was actually recorded as received at each step.

### C.4 Stock ledger (bin card equivalent)
Add a `StockMovement` model — append-only, one row per actual stock change (received, issued, transferred out, transferred in, adjusted, written off), with item reference, quantity delta, resulting balance, reference to the source transaction (PO id, transfer id, requisition id), timestamp, and user. This becomes the audit trail that today's mutable `quantity_in_stock` integer can't provide.
- Migrate existing stock-changing code paths (PO receiving, internal issue/requisition, and whatever else currently mutates `quantity_in_stock` directly) to also write a `StockMovement` row — the mutable quantity field can stay as a fast-read cache, but it should now be derivable from summing movements, and a reconciliation check (sum of movements == current quantity) should exist to catch drift.
- **Done when**: a test performs several stock-changing operations and confirms the movement ledger's running balance matches the item's `quantity_in_stock` at every step, and a reconciliation function correctly flags an artificially-introduced mismatch.

---

## PHASE D — Budget control — decision needed before building

### D.1 — DECISION NEEDED: budget/vote-head control scope
A real government LPO can't be raised against an unbudgeted line item. Before building this: does this facility currently track a budget/vote-head structure anywhere (even informally, outside this system), and if so, at what granularity (per department? per category of item? annual, quarterly?) Building a budget-check gate without knowing the real structure risks either blocking legitimate purchases or being too permissive to matter. Write the specifics needed to `DECISIONS_PENDING.md` and stop — this entire phase waits on the answer. Don't build a generic/guessed budget model in the meantime.

---

## PHASE E — Inter-facility transfer (depends on Phase B.2's answer)

Build this according to whichever model B.2 resolved to.

### E.1 Outward transfer
A `StockTransfer` model: source facility (this one), destination facility (a `Facility` record — real if B.2 resolved to shared-network, else just a reference record for an external institution), line items (item, quantity), status (`DISPATCHED` → `IN_TRANSIT` if that distinction matters → `RECEIVED` or `DISCREPANCY`), dispatch note reference/number, dispatched-by user, timestamp. Dispatching a transfer records a `StockMovement` (transfer-out) reducing this facility's stock immediately — goods leaving the building should leave the ledger, not wait for confirmation from the other end.
- **Done when**: a transfer can be created and dispatched, stock decrements immediately with a corresponding movement record, tested.

### E.2 Inward confirmation
Whoever receives a transfer (at this facility, receiving from another) confirms actual quantities received against what was dispatched — same partial-receipt logic as Phase C.3, since short/damaged-in-transit shipments are exactly as real a scenario for transfers as for supplier deliveries. Confirming a receipt records a `StockMovement` (transfer-in) increasing stock only by what was actually confirmed, and flags any variance from the dispatched quantity for review rather than silently accepting it.
- **Done when**: a test dispatches a transfer, confirms a partial receipt, and confirms the variance is recorded and visible, not silently dropped.

### E.3 Outbound-only recording (if a receiving facility isn't a system user)
If B.2 resolved to "this system won't ever see the receiving side's confirmation directly" for external institutions, still record the dispatch fully (E.1) and provide a manual "mark as confirmed by receiving facility" action with a free-text confirmation reference (e.g., a signed waybill number) rather than leaving dispatched transfers in permanent limbo.
- **Done when**: this manual confirmation path is tested and clearly distinguished in the UI from a same-network confirmed receipt (E.2), so nobody mistakes an unverified manual confirmation for a real two-sided reconciliation.

---

## PHASE F — Reporting

### F.1 Bin card / stock card view
A per-item view showing its full movement history from Phase C.4's ledger — the digital equivalent of a physical bin card, since that's the artifact an inspector or auditor will expect to see.

### F.2 Stock reconciliation report
A report surfacing any items where the ledger-derived balance and the cached `quantity_in_stock` have drifted (per C.4's reconciliation check), so this becomes something staff actually look at periodically rather than a check that only runs in tests.

---

## Report back

Standard evidence requirements per task, plus: a clear statement of which `DECISIONS_PENDING.md` items (B.2, D.1) are still open at the end of this session, and confirmation that Phase E was not started if B.2 wasn't answered in time.
