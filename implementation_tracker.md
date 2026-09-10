# HMIS Implementation Tracker
**Last Updated:** 2026-09-11
**Repository:** mathunjoroge/hospital
**Branch:** main

---

## 📊 Summary

| Metric | Count |
|--------|-------|
| Total Phases | 9 |
| Phases Complete | 0 |
| Phases In Progress | 2 (Phase 0, T3.x) |
| Phases Not Started | 7 |
| Critical Blockers | 2 (Billing listener regression, Theatre fixture) |
| Tests Passing | 378+ |
| Coverage | ~55% |

---

## 🚨 Active Blockers

| Blocker | Impact | Status | Owner |
|---------|--------|--------|-------|
| Billing event listener regression | Charges not synced for dispensing, theatre, ward, payments | 🔧 Fix applied — needs verification | Engineering Lead |
| T3.2 theatre test fixture | Tests fail with `TheatreProcedure(description=...)` | 🔧 Fix applied — needs verification | Engineering Lead |
| Mock data in database | Pollutes analytics for stakeholder demo | 🔧 Cleanup script created | Engineering Lead |

---

## ✅ Completed Work (T3.x Parallel Workstream)

| Task | Status | Commit | Notes |
|------|--------|--------|-------|
| T3.1: Telemedicine TELEHEALTH encounters | ✅ Complete | `0eb1bc6` | Session start → IN_CONSULTATION, complete → DISCHARGED |
| T3.3: Referral lifecycle | ✅ Complete | `ae376a6` | Source → REFERRED_OUT, target → new REFERRAL encounter |
| T3.4: MCH ANC encounters | ✅ Complete | `49fdc42` | ANC visit → encounter, immunization → encounter_id |
| T3.5: Ward daily charges cron | ✅ Complete | `cb8c973` | Midnight job, idempotent via source_id hash |
| T3.7: Pharmacy encounter scoping | ✅ Complete | `1b74259` | Dispensing blocked for DISCHARGED encounters |
| T3.8: Billing source_encounter_id | ✅ Complete | `cb8c973` | sync_charge accepts source_encounter_id param |
| T3.2: Theatre state machine | ⚠️ Partial | `bc83303` | PRE_OP→INTRA_OP→POST_OP wired, test fixture broken |
| Telemedicine WebRTC + SocketIO | ✅ Complete | `f42abc2a` | Full video calling with signaling server |
| Patient search (ID + Name) | ✅ Complete | `8107e111` | Mixed search with ilike |
| TLS/HSTS baseline | ✅ Complete | Sep 2026 | HSTS header shipped |
| Consent enforcement (DPA 2019) | ✅ Complete | Sep 2026 | Not in v1.0 scope |
| Referrals & appointments | ✅ Complete | Sep 2026 | Not in v1.0 scope |

---

## 🔧 In Progress

### Phase 0: Consolidation
| Item | Status | Notes |
|------|--------|-------|
| P0-01: MFA enforcement org-wide | ⏳ Not started | |
| P0-02: Backup restore test | ⏳ Not started | |
| P0-03: Celery dead-letter queue | ⏳ Not started | |
| P0-04: SIEM export wiring | ⏳ Not started | |
| P0-05: Break-glass quarterly review | ⏳ Not started | |
| P0-06: Extend encryption to lab/imaging/notes | ⏳ Not started | |
| P0-07: Full encounter lifecycle integration test | ✅ Complete | `test_full_encounter_lifecycle.py` |
| P0-08: OpenAPI annotations for FHIR | ⏳ Not started | |
| P0-09: Read replica configuration | ⏳ Not started | |
| P0-10: On-call rotation | ⏳ Not started | |

### T3.x Remaining
| Item | Status | Notes |
|------|--------|-------|
| T3.2: Fix theatre test fixture | 🔧 Fix applied | `TheatreProcedure(description=...)` → `type="General"` |
| T3.8: Restore full billing listener | 🔧 Fix applied | Two-phase + independent session + thread-safe |
| Mock data cleanup | 🔧 Script created | `scripts/cleanup_mock_data.py` |

---

## 📅 Upcoming Phases

### Phase 1: Terminology & CDSS (Weeks 1–12)
| Item | Status | Priority |
|------|--------|----------|
| P1-01: WHO ICD-API credentials | ⏳ Not started | High |
| P1-02: icd10_importer.py | 🔧 Script skeleton created | High |
| P1-03: Replace ICD10_DATABASE list | ⏳ Not started | High |
| P1-04: Alembic migration for icd10_codes | ⏳ Not started | High |
| P1-05: FHIR Condition ICD-10 system URL | ⏳ Not started | Medium |
| P1-06: SNOMED CT affiliate licence | ⏳ Not started | High |
| P1-07: snomed_importer.py | ⏳ Not started | High |
| P1-08: PatientProblem.snomed_code | ⏳ Not started | Medium |
| P1-09: SNOMED Procedure mapping | ⏳ Not started | Medium |
| P1-10: LOINC table creation | 🔧 Script skeleton created | High |
| P1-11: LabTest.loinc_code | ⏳ Not started | High |
| P1-12: FHIR DiagnosticReport LOINC | ⏳ Not started | Medium |
| P1-13: CDSS alert severity tiers | ⏳ Not started | High |
| P1-14: Pharmacist override workflow | ⏳ Not started | High |
| P1-15: P&T Committee dashboard | ⏳ Not started | Medium |
| P1-16: Cross-sensitivity rules | ⏳ Not started | High |

### Phase 2: Observability & DR (Weeks 6–18)
| Item | Status | Priority |
|------|--------|----------|
| P2-01: OpenTelemetry collector | ⏳ Not started | High |
| P2-02: Flask instrumentation | ⏳ Not started | High |
| P2-03: Sentry deployment | ⏳ Not started | High |
| P2-04: Uptime dashboard | ⏳ Not started | Medium |
| P2-05: SLO definitions | ⏳ Not started | High |
| P2-06: RPO/RTO definition | ⏳ Not started | High |
| P2-07: WAL archiving | ⏳ Not started | High |
| P2-08: DR failover script | ⏳ Not started | High |
| P2-09: Read replica | ⏳ Not started | High |
| P2-10: DICOM backup | ⏳ Not started | Medium |
| P2-11: TLS enforcement | ✅ Complete | Sep 2026 |
| P2-12: Secrets rotation | ⏳ Not started | High |
| P2-13: CSP headers + XSS suite | ⏳ Not started | High |
| P2-14: PostgreSQL RLS | ⏳ Not started | Medium |
| P2-15: Dependency audit | ⏳ Not started | Medium |

### Phase 3: Controlled Drug Register (Weeks 8–20)
| Item | Status | Priority |
|------|--------|----------|
| P3-01: ControlledDrug model | ⏳ Not started | Legal blocker |
| P3-02: ControlledDrugDispense model | ⏳ Not started | Legal blocker |
| P3-03: Dual-signature workflow | ⏳ Not started | Legal blocker |
| P3-04: Running balance ledger | ⏳ Not started | Legal blocker |
| P3-05: Shift reconciliation | ⏳ Not started | Legal blocker |
| P3-06: PPB-format register | ⏳ Not started | Legal blocker |
| P3-07: Tests | ⏳ Not started | Legal blocker |
| P3-08: Clinical sign-off | ⏳ Not started | Legal blocker |

### Phase 4: HL7 v2 MLLP (Weeks 14–28)
| Item | Status | Priority |
|------|--------|----------|
| P4-01: Interface engine selection | ⏳ Not started | High |
| P4-02: Mirth Connect deployment | ⏳ Not started | High |
| P4-03: HL7 result receiver | ⏳ Not started | High |
| P4-04: ADT sender | ⏳ Not started | Medium |
| P4-05: Analyzer instrumentation | ⏳ Not started | High |
| P4-06: LIS order workflow | ⏳ Not started | High |
| P4-07: Panic alert extension | ⏳ Not started | Medium |
| P4-08: MLLP load test | ⏳ Not started | Medium |

### Phase 5: PACS / DICOMweb (Weeks 20–32)
| Item | Status | Priority |
|------|--------|----------|
| P5-01: Orthanc deployment | ⏳ Not started | Medium |
| P5-02: OHIF Viewer integration | ⏳ Not started | Medium |
| P5-03: Imaging order to PACS | ⏳ Not started | Medium |
| P5-04: DICOM send from modality | ⏳ Not started | Medium |
| P5-05: FHIR ImagingStudy | ⏳ Not started | Medium |
| P5-06: DICOM metadata migration | ⏳ Not started | Low |
| P5-07: Orthanc object storage | ⏳ Not started | Medium |
| P5-08: Radiologist acceptance | ⏳ Not started | Medium |

### Phase 6: SMART on FHIR / SSO (Weeks 24–40)
| Item | Status | Priority |
|------|--------|----------|
| P6-01: Keycloak deployment | ⏳ Not started | Critical |
| P6-02: OAuth2 Bearer migration | ⏳ Not started | Critical |
| P6-03: SMART launch sequence | ⏳ Not started | Critical |
| P6-04: SMART scopes | ⏳ Not started | Critical |
| P6-05: CapabilityStatement | ⏳ Not started | Medium |
| P6-06: LDAP federation | ⏳ Not started | High |
| P6-07: SCIM provisioning | ⏳ Not started | Medium |
| P6-08: Session revocation | ⏳ Not started | High |
| P6-09: Patient portal SSO | ⏳ Not started | High |
| P6-10: Patient SMART launch | ⏳ Not started | Medium |

### Phase 7: Disease Programs (Weeks 32–52)
| Item | Status | Priority |
|------|--------|----------|
| P7-01 to P7-05: HIV/ART module | ⏳ Not started | Medium |
| P7-06 to P7-08: TB/DOTS module | ⏳ Not started | Medium |
| P7-09 to P7-11: Malaria module | ⏳ Not started | Medium |

### Phase 8: Pen Test + HIMSS (Weeks 44–56)
| Item | Status | Priority |
|------|--------|----------|
| P8-01 to P8-04: Penetration test | ⏳ Not started | Gate |
| P8-05 to P8-07: HIMSS Stage 6 | ⏳ Not started | Gate |
| P8-08 to P8-09: HIMSS Stage 7 | ⏳ Not started | Gate |

---

## 📝 Recent Commits

| Date | Commit | Description |
|------|--------|-------------|
| 2026-09-11 | `4bf58d05` | fix: T3.8 billing listener two-phase + independent session |
| 2026-09-11 | `c16c7a00` | fix: T3.8 billing listener plain value capture |
| 2026-09-11 | `8107e111` | feat: patient search ID + Name |
| 2026-09-11 | `f42abc2a` | feat: telemedicine WebRTC + SocketIO |
| Sep 10 | `bc83303` | feat: T3.2 theatre state machine |
| Sep 10 | `cb8c973` | feat: T3.5 ward charges + T3.8 billing scoping |
| Sep 10 | `49fdc42` | feat: T3.4 MCH ANC encounters |
| Sep 10 | `ae376a6` | feat: T3.3 referral lifecycle |
| Sep 10 | `0eb1bc6` | feat: T3.1 telemedicine encounters |
| Sep 10 | `1b74259` | feat: T3.7 pharmacy encounter scoping |

---

## 🔗 Related Documents

- [DECISIONS_PENDING.md](DECISIONS_PENDING.md) — Items requiring human sign-off
- [HMIS_WorldClass_Implementation_Plan_v1.1.md](HMIS_WorldClass_Implementation_Plan_v1.1.md) — Full roadmap
- [docs/backup_restore_runbook.md](docs/backup_restore_runbook.md) — Backup procedures
- [docs/security_audit_readiness.md](docs/security_audit_readiness.md) — Pen test scope

---

## 📌 Notes

- **Billing listener regression:** Force-pushed back to earlier draft. Fix applied with two-phase capture + independent session + thread-safe storage. Needs test verification.
- **Theatre fixture bug:** `TheatreProcedure(description=...)` fails because model has no `description` column. Fixed to use `type="General"`.
- **Mock data:** Cleanup script created at `scripts/cleanup_mock_data.py`. Run with `--dry-run` first.
- **Coverage:** Currently ~55%. Target 45% minimum (CI gate). No regression risk from current fixes.
