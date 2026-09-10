# HMIS Implementation Prompts — Claude Code

Nine self-contained prompts, one per phase of `HMIS_WorldClass_Implementation_Plan_v1.1.md`,
ordered by priority rather than by phase number. Run them one at a time, in order.
Review the diff and re-run the full test suite before starting the next one.

## Why not one giant prompt

A single prompt covering all 327 person-days of work produces one huge diff you can't
meaningfully review, loses track of earlier decisions by the time it reaches later phases,
and can't stop for the two phases that need a human/legal/clinical decision before any code
should exist. This repo has already shown what happens when a large unattended change lands
without a review gate — see 01 below.

## Run order and why

| # | File | Phase | Why it's here |
|---|------|-------|----------------|
| 1 | `01_immediate_fixes.md` | Phase 0 cleanup | Actively broken right now — billing sync silently lost most of its coverage, a test has a bad fixture. Fix before building anything on top. |
| 2 | `02_controlled_drug_register.md` | Phase 3 | Legal blocker (Kenya PPB regulation) for a hospital already dispensing Schedule II/IV drugs. Highest real-world exposure — but it starts with a decision-drafting task, not code, since the dual-signatory policy must be resolved first. |
| 3 | `03_security_observability_dr.md` | Phase 2 | High priority, and P2-14 (row-level security) is an explicit prerequisite for Phase 6 — do this before SSO, not after. |
| 4 | `04_sso_smart_on_fhir.md` | Phase 6 | The plan's own "Critical" label and, in its words, "the single most important gap for Tier 1 hospital interoperability." Depends on #3 being done first. |
| 5 | `05_terminology_cdss_hardening.md` | Phase 1 | High priority, mostly independent of the others — can be reordered earlier if you want it running in parallel with a second engineer. |
| 6 | `06_hl7v2_mllp_interface.md` | Phase 4 | High priority — without it, every lab result is manual data entry. |
| 7 | `07_national_disease_programs.md` | Phase 7 | Medium priority, decision-gated (clinical review board) like Phase 3 — decision-drafting first, code after sign-off. |
| 8 | `08_pacs_dicomweb.md` | Phase 5 | Medium priority. |
| 9 | `09_pentest_himss_gate.md` | Phase 8 | Gate phase — mostly external-firm coordination, not code. Must run last; needs 1–8 substantially done first. |

## Rules that apply to every prompt (also repeated in each file)

- Work on a new branch per phase (e.g. `feat/phase0-cleanup`). **Never commit to `main`
  directly. Never force-push, on any branch.** This repo has already had a force-push
  silently delete most of the billing sync logic once — don't repeat it.
- Before starting, run the full test suite and note the baseline pass count. If baseline is
  already red, stop and report — don't build on top of a broken baseline.
- Touch only files in scope for this phase. No incidental refactors, renames, or "cleanup"
  of code outside the listed work items, even if it looks related.
- Every acceptance criterion in the work-item table needs a test. An item isn't done without
  one.
- After each work item, re-run the *full* suite (not just the new tests) and
  `ruff check . --select E,F,W,I --ignore E501` — both must stay clean.
- Finish by leaving the branch ready for a PR: a summary mapping commits to work-item IDs
  (e.g. P0-01, P2-07) and the test-count/coverage delta vs. baseline. Do not merge it yourself.
- If a work item's acceptance criteria depend on a decision that isn't recorded in
  `DECISIONS_PENDING.md`, stop and ask — don't assume an answer and code around it.
