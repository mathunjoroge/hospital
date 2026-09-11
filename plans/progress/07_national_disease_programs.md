# Prompt 7 of 9 — National Disease Program Modules: HIV/ART, TB, Malaria (Phase 7)

Priority: Medium. Decision-gated like Phase 3 — a clinical reviewer must sign off each data
model before it's built.

## Stop condition — read this before doing anything else

Three decisions in `DECISIONS_PENDING.md` item 11 are **not yet resolved**:
1. Named clinical reviewer per program (HIV clinician, TB clinician, malaria specialist) who
   signs off the data model and reporting fields.
2. Scope: are all three programs in scope for initial deployment, or only the ones relevant
   to the target facility's patient population?
3. Whether each program ties into the existing DHIS2 exporter from day one, or is built
   standalone first.

**Do not invent a data model for any of these three programs without a named clinical
reviewer attached to it.** The existing MCH/ANC module is the reference depth to match, but
copying its structure is not a substitute for clinical sign-off on program-specific fields
(ART regimen lines, TB drug sensitivity, malaria species/parasite density) — these carry real
clinical-safety weight.

## Part A — Decision drafting (do this first)

Add `docs/decisions/phase7_national_programs_proposal.md`:
- For each of HIV/ART, TB, Malaria: what data model shape is proposed (using the MCH/ANC
  module as a structural reference), what open clinical questions it has, and who the
  proposed reviewer is if known.
- A recommendation on scope (all three vs. facility-relevant subset) with reasoning, left as
  a decision for the human to make, not decided by the agent.
- A recommendation on DHIS2 integration timing (day-one vs. standalone-first).

Stop here and wait for decisions to be recorded before Part B, per-program, as each is
resolved — the three programs don't need to unblock together.

## Part B — Implementation (per program, only after that program's reviewer/scope are resolved)

Branch: `feat/phase7-national-programs`. Never commit to `main` directly, never force-push.

**7A — HIV/ART**

| Item | Acceptance criteria |
|------|---------------------|
| P7-01 | Data model with the named HIV clinician: `ARTEnrollment`, `ARTRegimen` (line 1/2/3), `AdherenceVisit`, `ViralLoad`, `CD4Count`, `WHOStage`. Review against the MOH HTS/ART register format. |
| P7-02 | `ARTEnrollment` workflow: link to patient, enrolment date, unique ART number, baseline CD4/WHO stage. Prevent duplicate enrolments. |
| P7-03 | Regimen tracking against the MOH Kenya ART formulary (store as a DB table, not code constants, so a non-developer can update it — add a formulary change log). Flag unsupported combinations. Second-line switch requires a clinical reason + approval. |
| P7-04 | Adherence visit: date, pills dispensed/returned, calculated adherence %, category (good/fair/poor). Flag missed visits after 7 days via a Celery task. |
| P7-05 | Wire to the DHIS2 exporter: MOH 731 ART cohort report. Monthly automated export. |

**7B — TB/DOTS**

| Item | Acceptance criteria |
|------|---------------------|
| P7-06 | Data model with the named TB clinician: `TBEnrollment`, `DOTSPhase` (intensive/continuation), `SputumResult`, `XpertResult`, `DrugSensitivity`, `TreatmentOutcome`. |
| P7-07 | DOTS adherence tracker: daily observed-therapy record. Design the CHW-facing daily tick as an SMS confirmation flow first — do not assume CHWs have smartphones. Flag 2+ consecutive missed days. |
| P7-08 | Wire to DHIS2 exporter: MOH TB register. Quarterly export. |

**7C — Malaria**

| Item | Acceptance criteria |
|------|---------------------|
| P7-09 | Data model: `MalariaTest` (RDT/microscopy), species (P. falciparum/vivax/malariae), parasite density, treatment prescribed, outcome. |
| P7-10 | Wire to DHIS2 exporter: MOH 705A/B malaria cases. Monthly export (the exporter partially exists in `dhis2_exporter.py` — extend it, don't rebuild). |
| P7-11 | Outbreak signal: if malaria cases in any 7-day window exceed 2× the 4-week rolling average, generate a public health alert to the facility medical officer. |

## Done when (per program)

- Full test suite passes; ruff clean.
- That program's clinical reviewer sign-off is recorded or explicitly flagged as pending in
  the PR — not assumed.
- Branch pushed (not force-pushed), ready for PR.
