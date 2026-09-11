# Prompt 5 of 9 — Terminology Completion & CDSS Hardening (Phase 1)

Priority: High. Mostly independent of the other phases — can be run in parallel with a
second engineer if you have one.

Branch: `feat/phase1-terminology-cdss`. Never commit to `main` directly, never force-push.
Run the full test suite before starting and record the baseline.

## 1A — Full ICD-10-CM/PCS ingestion

| Item | Acceptance criteria |
|------|---------------------|
| P1-01 | WHO ICD-API credentials need to exist and be recorded in `DECISIONS_PENDING.md` item 4 before this starts. If they're not there, stop and ask rather than hardcoding placeholder credentials. |
| P1-02 | Write `icd10_importer.py`: paginate the WHO API, store code + description + chapter + block in an `icd10_codes` table with a GIN full-text index. |
| P1-03 | Replace the `ICD10_DATABASE` list in `departments/medicine/prescribe.py` with a DB query against `icd10_codes`. Maintain backward compatibility for existing coded records — do not silently remap existing codes. |
| P1-04 | Alembic migration for `icd10_codes`. Seed migration runs the importer. Add to the docker-compose entrypoint. |
| P1-05 | FHIR R4 `Condition` endpoint returns the WHO system URL (`http://hl7.org/fhir/sid/icd-10`) in `coding.system`. |

If WHO credentials are delayed, the fallback is the free NLM UMLS ICD-10 flat file with the
same schema — use that rather than blocking, but note it clearly in the PR as a temporary
substitute pending the WHO credentials.

## 1B — SNOMED CT for problem list and procedures

| Item | Acceptance criteria |
|------|---------------------|
| P1-06 | SNOMED CT affiliate licence application needs to be recorded in `DECISIONS_PENDING.md`. If licensing status is unclear, start with the freely available SNOMED CT CORE subset (~10,000 most-used concepts) rather than blocking on the full licence. |
| P1-07 | Write `snomed_importer.py` to load concept + description + relationship tables from the SNOMED CT International Release RF2 snapshot (~350k rows). |
| P1-08 | Add `PatientProblem.snomed_code` column. Problem-list UI gets SNOMED typeahead search. Validate the concept is a Clinical Finding (hierarchy `404684003`). |
| P1-09 | Map procedure orders to the SNOMED Procedure hierarchy (`71388002`). Expose via FHIR R4 `Procedure` resource. |

## 1C — Full LOINC for laboratory

| Item | Acceptance criteria |
|------|---------------------|
| P1-10 | Download LOINC Table Core CSV, load into `loinc_codes` (`LOINC_NUM`, `LONG_COMMON_NAME`, `COMPONENT`, `SYSTEM`). |
| P1-11 | Add `loinc_code` column to `LabTest`. Seed from `loinc_codes` for the existing ~200-test catalog. Require `loinc_code` on new test creation going forward. |
| P1-12 | FHIR R4 `DiagnosticReport` and `Observation` endpoints include LOINC codes in `coding[]`. |

## 1D — CDSS alert fatigue tuning

This subsection changes clinical safety behavior — treat it with more caution than the
terminology work above.

| Item | Acceptance criteria |
|------|---------------------|
| P1-13 | Introduce alert severity tiers on `SafetyAlert`: CRITICAL (absolute contraindication, cannot override), HIGH (pharmacist approval required), MODERATE (warn + document), LOW (informational, not surfaced in prescribing UI). |
| P1-14 | Pharmacist override workflow for HIGH-tier alerts: prescriber flags, pharmacist reviews within 2 hours, approval logged to `AuditLog`. |
| P1-15 | P&T Committee alert review dashboard: override frequency by drug pair, accept/dismiss ratios, time-to-review. Monthly PDF export. |
| P1-16 | Cross-sensitivity rules: Penicillin→Cephalosporins (10% cross-react), Sulfonamides→Furosemide, NSAIDs→Aspirin. Wire into the existing allergy engine. |

**Before deploying any tier reclassification (P1-13), run it in shadow mode — log but do not
suppress alerts — for two weeks, and get explicit Clinical Lead sign-off before switching
suppression on for real. Do not enable suppression as part of the same change that introduces
the tiers.** Reclassifying existing alerts can silently remove ones clinicians currently rely
on; this needs a human clinical review, not just test coverage.

## Done when

- Full test suite passes; ruff clean.
- P1-13's shadow-mode requirement is respected — suppression is not turned on without a
  recorded Clinical Lead sign-off.
- Branch pushed (not force-pushed), ready for PR.
