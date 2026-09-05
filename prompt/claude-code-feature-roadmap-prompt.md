# Hospital HMIS — Feature Completion Roadmap

## How to use this document

Same model as the hardening briefing: paste the "Mission briefing" section as your first message to Claude Code, work through phases in order, and keep updating a tracker file as you go. This is new feature work on top of an already-hardened codebase — it should not touch the security/compliance work from the previous plan except where a new feature needs to plug into it (e.g., new AI call sites must go through the consent gate; anything touching patient PII must respect `EncryptedString`).

---

## Mission briefing (paste this to Claude Code)

You're extending a hardened Flask 3.0 hospital management system (12 department blueprints, a separate FastAPI NLP microservice, Postgres, Redis, real RBAC/CSRF/audit-logging/encryption infrastructure already in place). A review found the system is staff-facing only — patients have no self-service access, there's no outbound communication channel (SMS/push), no hospital-wide analytics view, no emergency-access pattern, and clinical decision support (drug interactions) is minimal. This document scopes the work to close those gaps.

### Ground rules (same as before, still non-negotiable)

1. **Branch per phase**: `features/phase-a`, `features/phase-b`, etc. Never commit directly to `main`.
2. **Test before and after.** No behavior change without a test.
3. **Flag judgment calls, don't invent them.** Several phases below have explicit `DECISION NEEDED` items — these involve cost (SMS gateway fees), legal/regulatory exposure (telemedicine, patient data access), or product direction that isn't yours to decide. Scaffold around them, don't guess through them.
4. **Patient data is never destructively migrated.** Same as before.
5. **New PII gets encrypted from the start.** Any new column holding patient-identifying data uses `EncryptedString` from day one — don't repeat the gap where it existed but wasn't applied.
6. **New AI call sites go through the consent gate.** If a phase below adds any new call to an external AI API involving patient data, it must use the `has_ai_consent()` check from the hardening follow-up, not reinvent its own.
7. **Update the tracker.** Extend the existing `IMPLEMENTATION_PROGRESS.md` with a new `## FEATURE ROADMAP` section mirroring this document's phases, rather than starting a separate file.

---

## PHASE A — Patient self-service portal

Goal: patients can do something themselves instead of everything requiring a staff member.

### A.1 Decide the access model — DECISION NEEDED, ask before building
Before any code: how does a patient authenticate? Options with real tradeoffs — a full username/password account (highest friction, most secure), OTP-via-SMS/email tied to a phone number or national ID already on file (lower friction, depends on Phase B's SMS work existing first), or a scoped magic-link per visit (lowest friction, weakest for returning multi-visit use). Also decide: does this need a separate `PatientUser` identity distinct from the clinical `Patient` record (recommended — don't let the portal login double as the clinical record), and what can a patient actually see (own results only, or also billing/insurance status)? Do not proceed past this task until these are answered.

### A.2 Patient authentication
Implement whatever was decided in A.1. Use the existing `Flask-Limiter`/lockout patterns for this new login surface too — don't leave the new patient-facing login unprotected while the staff one is hardened.

### A.3 Patient-facing views (read-only first)
- Own upcoming/past appointments
- Own lab results (once released/verified by staff — don't expose a result before a clinician has reviewed it; check whether `RequestedLab`/`LabTest` already has a "verified" or "released" status flag to gate on, and add one if not)
- Own billing/invoice status and payment history
- Own insurance/claim status (from the SHA/SHIF module)

### A.4 Patient-initiated actions (write, more caution)
- Appointment request/booking against available slots (needs a real slot/schedule model — check if one exists under `records` or `nursing`; if not, this is its own sub-task, flag if it turns out to be bigger than expected)
- Updating own contact details (with the update itself going through the existing audit-logging pattern, attributed to the patient's own portal identity, not a staff user)

**Done when**: a patient can log in (via whatever A.1 decided), see their own data and nothing else, and a test proves patient A cannot see patient B's records through this surface under any of the new routes.

---

## PHASE B — Outbound patient communication (SMS/notifications)

Goal: the system can actually reach a patient, not just staff.

### B.1 Decide the SMS gateway — DECISION NEEDED, ask before building
This has a real recurring cost. Africa's Talking is the common choice for Kenya-based deployments (local rates, good documentation); Twilio is more expensive here but more globally standard. Ask which (or whether SMS is even in budget right now vs. starting with email-only via the existing `Flask-Mail` setup) before writing integration code.

### B.2 Notification event framework
Extend the existing `Notifications` model (currently staff-only, in-app) into a real event-driven system: define a small set of triggerable events (appointment reminder, lab result ready, invoice due, claim status changed) and a delivery-channel abstraction (in-app / SMS / email) so a new event type doesn't require touching delivery code, and a new channel doesn't require touching every event's business logic.

### B.3 Wire real triggers
- Appointment reminder: scheduled job (there's already `Flask-APScheduler` in the dependencies — use it) X hours before an appointment.
- Lab result ready: trigger when a result's status flips to released (ties into A.3's release-gating work).
- Invoice due / payment received: hook into the `Invoice`/`Payment` models from the unified billing work.

**Done when**: triggering each event in a test sends a real (sandbox, if SMS) message and logs the delivery attempt and outcome — including failures, which shouldn't silently disappear.

---

## PHASE C — Hospital-wide analytics dashboard

Goal: an administrator can see the hospital's state at a glance, not just per-department reports.

### C.1 Decide the KPI set — DECISION NEEDED, ask before building
Don't invent what matters to a hospital administrator. Common candidates: bed occupancy rate, average patient wait time, revenue vs. period, claims pending/approved/rejected ratio, top diagnoses by volume, drug stockout incidents. Confirm which of these (plus anything else) actually matter before building the queries — pulling the wrong metrics wastes the work and clutters the dashboard.

### C.2 Build the aggregation layer
Write the actual queries/aggregations per confirmed KPI, likely as a `departments/admin/analytics.py` service module separate from the route handlers, so the underlying numbers can be tested independent of rendering.

### C.3 Dashboard view
A new admin-only route/template rendering the confirmed KPIs, using the existing chart-rendering approach already in the codebase (check what's already used for nursing vitals charts) for consistency rather than introducing a new charting library.

**Done when**: the dashboard renders real numbers computed from the actual database (verified against a manually-computed expected value in a test), refreshes on a reasonable schedule (doesn't need to be real-time), and is gated behind `roles_required('admin')`.

---

## PHASE D — Break-glass emergency access

Goal: a clinician facing a genuine emergency isn't blocked by normal RBAC, but that override is loud, not silent.

### D.1 Decide what qualifies — DECISION NEEDED, ask before building
Break-glass access is a real clinical-safety and liability question, not a pure engineering one: which roles can invoke it, does it require a stated reason at the point of use, does it require a second person's sign-off (even after the fact), and what's the maximum scope/duration of the override? Don't build a default policy here — ask.

### D.2 Implementation
Once scoped: add an override path that bypasses the normal `roles_required` check for a specific, logged reason, tied to the user's identity, with:
- A mandatory reason field captured at the point of use
- Immediate, highly visible audit logging (distinct log category from normal audit entries, not just another row in the same table — this should be trivially queryable/reportable on its own)
- An automatic notification to a supervisor/compliance role (ties into Phase B's notification framework) whenever break-glass is invoked
- Time-boxing: the override expires and reverts automatically, it isn't a permanent permission change

**Done when**: a test invokes break-glass access, confirms access is granted despite normal RBAC denying it, confirms the distinct audit trail entry exists with the reason captured, and confirms it expires correctly.

---

## PHASE E — Expand drug-interaction checking

Goal: replace the current small hardcoded interaction/allergy list with something backed by real data.

### E.1 Extend the existing DrugCentral connection
The app already has a working DrugCentral Postgres connection (used for both the medicine drug-reference pages and the AI Discovery similarity search). DrugCentral includes structured interaction/pharmacology data — investigate what's actually available in the schema (don't assume; inspect the live schema, similar to how the AI Discovery similarity search work was scoped) before committing to an approach.
### E.2 Replace or supplement the hardcoded list
If DrugCentral's data is usable for this: replace `KNOWN_INTERACTIONS`/`ALLERGY_GROUPS` in `departments/medicine/prescribe.py` with real lookups, keeping the existing hardcoded list as a fallback if the DB is unreachable (same graceful-degradation principle used elsewhere in this codebase) rather than a hard dependency.
If DrugCentral's interaction data turns out to be too sparse or unstructured to use directly: say so and propose an alternative (a maintained open dataset, or scoping this as "expand the hardcoded list carefully with more entries" rather than a full data-backed system) rather than forcing a bad integration.

**Done when**: prescribing a drug that interacts with an existing DrugCentral-sourced interaction (not just the 5 originally hardcoded pairs) triggers the same alert UI, and the fallback path is tested by simulating a DB-unavailable condition.

---

## PHASE F — Lab instrument / imaging integration (discovery only)

Goal: figure out what's actually feasible before committing engineering time neither of us can validate without real hardware.

This phase is a **written recommendation, not code** — same treatment as the offline-first discovery task in the hardening plan. Research and report back on:
- What lab analyzer interfacing (HL7 v2 ORU/ORM messaging is the common standard) would actually require, and whether it's realistic without access to real lab hardware/vendor documentation for this facility's actual equipment.
- What a genuinely useful next step for imaging looks like beyond DICOM metadata handling — a basic in-browser DICOM viewer (open-source options exist, e.g. Cornerstone.js) is achievable without hardware integration and might be the better near-term target than full PACS interfacing.

**Done when**: a written recommendation exists, reviewed before any implementation begins.

---

## PHASE G — Telemedicine — DECISION NEEDED before this phase is even scoped

Don't start this phase. Video consultation brings real-time infrastructure cost, provider licensing/telehealth-regulatory questions (Kenya-specific rules on remote consultation may apply), and a substantially different engineering surface (WebRTC, TURN/STUN infra, recording/consent-for-recording questions) than anything else in this roadmap. Flag back to the human whether this is actually wanted before any exploration work happens — this is explicitly out of scope until that conversation happens.

---

## Final note

As with the hardening plan: completing any phase here is not authorization to treat the system as ready for real patients. That sign-off is a human decision, informed by both this roadmap and the hardening plan's own final note.
