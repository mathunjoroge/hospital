# Prompt 6 of 9 — HL7 v2 MLLP Interface Engine (Phase 4)

Priority: High. Without this, every lab result and ADT event requires manual data entry.

Branch: `feat/phase4-hl7v2-mllp`. Never commit to `main` directly, never force-push. Run the
full test suite before starting and record the baseline.

## Architecture decision — confirm before building

Deploy a dedicated HL7 interface engine as a separate service alongside the Flask app
(Mirth Connect / NextGen Connect is the open-source recommendation; Rhapsody is the
commercial alternative). The HMIS is the "system of record" receiving transformed FHIR
resources from the interface engine — it does not parse raw HL7 v2 itself.

| Item | Acceptance criteria |
|------|---------------------|
| P4-01 | Interface engine choice (Mirth vs Rhapsody) needs to be recorded in `DECISIONS_PENDING.md`, including estimated licensing cost and support model. If not recorded, default to Mirth Connect (free, self-hosted) and flag the choice in the PR for confirmation rather than silently committing to a paid option. |
| P4-02 | Add a `mirth-connect` container to `docker-compose.yml`. Configure channels: `ORM^O01` (lab order outbound), `ORU^R01` (result inbound), `ADT^A01/A03/A08` (admit/discharge/update). |
| P4-03 | Build the HMIS HL7 result receiver: `POST /api/hl7/oru` accepts transformed FHIR `DiagnosticReport` from Mirth. Updates `LabResult`, triggers the existing LIS panic alert on critical values. |
| P4-04 | Build the HMIS ADT sender: on `Patient.create/update/admit/discharge`, publish an ADT FHIR message to the Mirth outbound channel. |
| P4-05 | Per-analyzer Mirth channel (vendor HL7 dialects differ). Test against at least the Roche Cobas and Sysmex XN-series HL7 simulators before assuming any other analyzer works from the same channel config. |
| P4-06 | End-to-end LIS order workflow: clinician creates lab order → `ORM^O01` sent to analyzer → analyzer runs test → `ORU^R01` received → result appears in HMIS within 2 minutes. |
| P4-07 | Extend the existing panic alert workflow: LIS panic threshold trigger fires immediate SMS + in-app push to the ordering clinician. |
| P4-08 | Load test the MLLP connection at 50 concurrent ORU messages. Verify no message loss with Mirth persistence enabled. |

## Risks to handle explicitly

- Every analyzer vendor's HL7 v2 dialect is slightly non-standard — don't assume one
  channel config works for a second analyzer model without testing against its own
  conformance statement.
- Hospital LAN connections drop long-lived TCP sessions frequently — configure Mirth
  keepalive and reconnect-on-failure, and wire connection-drop alerts into the Phase 2
  observability stack rather than failing silently.
- Profile the 2-minute result-delivery SLA against real message volume before treating it as
  guaranteed — MLLP itself is synchronous and the bottleneck is usually the Mirth transform
  step, not the network.

## Done when

- Full test suite passes; ruff clean.
- The end-to-end order → result loop (P4-06) is demonstrated against at least one analyzer
  simulator, not just unit-tested in isolation.
- Branch pushed (not force-pushed), ready for PR.
