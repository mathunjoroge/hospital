# Hospital HMIS — Path to World-Class

## How to use this document

Paste the "Mission briefing" section as your first message to Claude Code. Read the **Process Integrity** section first, before any phase — it exists because five separate `DECISION NEEDED` gates in prior work on this repo were overridden instead of honored, including one (telemedicine) that was explicitly forbidden pending human input. This document assumes that pattern will repeat unless the process itself, not just the wording, changes. If you're Claude Code reading this: the instructions in Process Integrity are not suggestions to weigh against finishing the work faster. They are the actual deliverable of this session, as much as any code is.

---

## Mission briefing (paste this to Claude Code)

You are continuing work on a hardened Flask 3.0 hospital management system. Prior sessions delivered real, verified security fixes (RBAC, CSRF, encryption at rest, audit logging, login lockout) and real features (patient portal, SMS notifications, analytics dashboard, break-glass access). They also, independently and repeatedly, ignored explicit instructions to stop and ask a human before making judgment calls with legal, financial, or safety consequences — including building a telemedicine feature that was explicitly marked out of scope pending a decision that was never sought.

This session has two kinds of work: closing known technical debt, and — more importantly — operating under a process that doesn't depend on your judgment about when a rule matters.

---

## PROCESS INTEGRITY (read first, applies to everything below)

### P.1 — Hard stops are hard stops
Every `DECISION NEEDED` item below means: write what you'd need to proceed (the scaffolding, the options, the tradeoffs) into `DECISIONS_PENDING.md` at the repo root, then **end your turn without writing the code that depends on the answer**. Do not pick the most reasonable-sounding option and continue in the same session, even with a comment flagging it as a placeholder. A comment saying `# TODO: confirm with human` next to code that already assumes an answer is not a hard stop — it's the same failure with better documentation. The test is: could a human read `DECISIONS_PENDING.md`, do nothing else, and know that no downstream code exists yet that depends on their answer? If not, it wasn't a real stop.

### P.2 — Evidence over claims
Every task's "done" report must include the actual command you ran and its actual output — a real database row, a real HTTP response, a real test name and its result — not a summary sentence like "tests passing" or "implemented successfully." If you cannot produce direct evidence for a claim, say so explicitly rather than asserting completion. Prior sessions reported phases complete that, on inspection, were unstarted, partially done, or built as a different (sometimes explicitly forbidden) feature under the same name. Assume the human will independently re-verify every claim in this session the same way — because they will.

### P.3 — No self-certification of "production-ready" or "world-class"
Nothing in this document, however thoroughly completed, authorizes a claim that this system is ready for real patients, or that it has reached "world-class" status. Several of the gaps below (external security audit, clinical safety review, accessibility audit, regulatory/compliance sign-off) are things you cannot do yourself — you can prepare for them, document against them, and make the system easier for a qualified human/external party to assess, but you cannot perform or substitute for them. If asked to assess overall readiness, say plainly which parts of that assessment are and aren't yours to make.

### P.4 — Branch and PR discipline, no exceptions
Every phase gets its own branch and a PR with a description containing a "Verification Evidence" section (per P.2). Do not merge to `main` yourself — open the PR and stop. This is a structural check, not a trust exercise: a PR sitting open and unmerged is a real, visible signal that something is waiting on human review, in a way that a chat message summarizing "what I did" is not.

---

## PHASE 1 — Close out existing debt

### 1.1 Telemedicine: quarantine, don't just fix
The existing `departments/telemedicine/` module was built despite being explicitly out of scope pending a decision that was never sought, and doesn't function as its own commit message claimed (no real WebRTC signaling exists despite the description). Do not complete or fix it into a working feature.
- Move it behind a hard feature flag that defaults to **off** (`TELEMEDICINE_ENABLED` env var, checked at the blueprint-registration level so the routes don't even load when unset) — don't just hide the nav link.
- Write a short note in `DECISIONS_PENDING.md`: telemedicine involves real-time infrastructure cost and telehealth regulatory questions specific to Kenya that need an explicit human decision before any further work — including whether to keep the existing scaffolding at all or remove it outright.
- **Done when**: the routes are unreachable with the flag unset (test this), and the decision is logged, not resolved.

### 1.2 Finish the AI consent gate
`has_ai_consent()` is only checked in `departments/medicine/chat_bot.py`. Extend it to every other call site sending patient-specific clinical data to NVIDIA or Gemini: `departments/nlp/src/nvidia_client.py`, `departments/medicine/oncology.py`, `departments/nlp/summarizer.py`, `departments/nlp/chatbot.py`, `departments/pharmacy/ai_discovery.py`, and any other file confirmed (via `grep -rln "NvidiaNIMClient\|gemini_api_key" --include="*.py" .`) to make an actual outbound call with patient data, as opposed to merely importing something that does. For each, add a test proving the call is refused without consent and proceeds with it, per the pattern already used in `chat_bot.py`.
- **Done when**: every real call site has a passing pair of tests, and you can `grep -c "has_ai_consent" <file>` each one and show a non-zero result with actual call-site context, not just an import line.

### 1.3 Real drug-interaction data (third attempt — investigate before building this time)
Two prior attempts at this produced a hardcoded rule list under two different names (`KNOWN_INTERACTIONS`, then `cdss.py`'s rule set) instead of the DrugCentral-backed system originally requested. Before writing any code:
1. Connect to the live DrugCentral database (credentials already in `departments/shared/drugcentral.py`) and actually inspect what interaction/pharmacology-relevant tables and columns exist — paste the real schema output, not an assumption about what DrugCentral "should" have.
2. If usable interaction data exists: build the lookup against it, with the existing hardcoded list as a documented fallback only for when the DB is unreachable.
3. If it doesn't (DrugCentral's public schema may not include structured DDI data — this is a real possibility, not a reason to fabricate a query against a table that doesn't exist): say so explicitly, and propose a specific named alternative (e.g., a specific open dataset with a real URL) rather than defaulting to "expand the hardcoded list" without saying that's what happened and why.
- **Done when**: either a real query against real DrugCentral tables is shown working with real output, or a clear written explanation of why that's not feasible exists — not a third hardcoded list presented as if it were data-backed.

### 1.4 Pharmacy PO/Supplier system — decision needed
This was built without being requested in any prior scoping document. Per P.1: don't extend or polish it. Write a `DECISIONS_PENDING.md` entry describing what exists and stop — whether to keep, remove, or formally scope it is a human call.

### 1.5 Phase F (lab/PACS): actually do the research this time
This was previously supposed to be a written recommendation and instead nothing happened at all. Produce the actual document: what HL7 v2 lab-instrument interfacing would require for this facility, and whether a basic browser-based DICOM viewer (e.g., Cornerstone.js) is a more realistic near-term target than full PACS integration. This is a markdown file, not code.

---

## PHASE 2 — Operational maturity

### 2.1 Real observability
`/healthz` exists but there's no error tracking or metrics. Add structured logging with a real sink (not just stdout) and integrate a real error-tracking service — Sentry has a generous free tier and a simple Flask integration; use it unless `DECISIONS_PENDING.md` needs to flag a cost concern first (check: does Sentry's free tier actually cover this app's expected volume, or does that need a human check on account/budget?).
- **Done when**: a deliberately triggered exception in a test environment appears in the error tracker, with evidence (a screenshot description or API confirmation isn't enough — paste the actual event ID/API response confirming it was received).

### 2.2 Make the CI security gate real
`pip-audit` in `.github/workflows/ci.yml` has `|| true`, so it can never fail the build. Remove that. For any currently-flagged vulnerability that's a genuine false positive or accepted risk, add a specific `--ignore-vuln <ID>` with a one-line comment explaining why — don't blanket-ignore everything to make the gate pass trivially.
- **Done when**: a deliberately reintroduced known-vulnerable dependency in a test branch causes the CI check to fail, demonstrated with an actual failed run, not a description of expected behavior.

### 2.3 Backup restore, actually tested
`scripts/backup_db.py` exists; there's no evidence a restore from one of its backups has ever been performed. Actually do it: take a real backup, actually tear down and restore a database from it, and document the exact steps and timing as a runbook. A backup nobody has restored from is not a backup.
- **Done when**: the runbook exists and you've personally executed it once against a non-production database, with the actual commands and output included.

### 2.4 Load/stress baseline
There's no evidence of load testing anywhere. Run a basic load test (Locust or k6, whichever is simpler to add) against the 3-5 highest-traffic-likely endpoints (login, patient search, appointment booking) and document actual results — requests/sec, error rate, p95 latency — at a couple of concurrency levels. This doesn't need to be exhaustive; it needs to exist as a real, honest baseline rather than an assumption that the app scales.
- **Done when**: a results file with real numbers exists, and you've noted anything that failed or degraded badly, not just the happy path.

---

## PHASE 3 — Preparing for external validation (not performing it)

This phase produces documentation and checklists for a human or third party to act on — it does not claim any of these processes are complete.

### 3.1 Security audit readiness
Produce a document summarizing the current security posture (what's implemented: RBAC, CSRF, encryption, audit logging, rate limiting) and a scoped list of what an external penetration test should specifically target given this app's architecture — don't perform the pentest yourself or claim one occurred.

### 3.2 Accessibility baseline
Run an automated accessibility scan (axe-core or similar) against the main patient-facing and staff-facing pages, fix what's mechanically fixable (missing `alt` text, contrast issues, missing form labels — some of this was already flagged in an earlier UI review; check whether it was actually fixed), and produce a report of what requires manual/expert review beyond automated scanning.

### 3.3 Clinical safety review packaging
Produce a document a clinical reviewer (not you) would need: what clinical decision support exists (drug interactions, allergy checks), what its actual scope and limitations are (be honest about the gap between "drug interaction checking exists" and "comprehensive clinical decision support"), and what questions a clinical safety reviewer should be asking. This is scoping the conversation, not having it on the reviewer's behalf.

---

## PHASE 4 — Final report

Do not summarize this session as "world-class achieved" or "production-ready." Instead, produce:
- A completed-vs-pending breakdown against every task above, each with its actual evidence inline (or a link to where it lives)
- The full, current contents of `DECISIONS_PENDING.md`
- An explicit statement of which remaining gaps require a human, a paid third party, or a policy decision — and which are still yours to close with more engineering time

## Final note

Same as every prior version of this note, restated because it keeps needing to be: completing this document is not authorization to treat the system as ready for real patients. That determination is a human one, and per Phase 3, parts of it aren't even a determination any single person can make alone — it needs external, qualified review this document cannot substitute for.
