# Prompt 4 of 9 — SMART on FHIR, OAuth2 & Enterprise Identity / SSO (Phase 6)

Priority: Critical — the plan's own text calls this "the single most important gap for Tier 1
hospital interoperability." Without it, the HMIS cannot exchange data with Epic, Cerner,
Meditech, or any major EHR.

## Prerequisite check — do this before anything else

Confirm `03_security_observability_dr.md` (Phase 2) is merged, specifically P2-14
(PostgreSQL row-level security). This phase adds multi-facility SSO on top of it; if RLS
isn't in place, stop and say so rather than building SSO on a data-isolation gap.

Branch: `feat/phase6-sso-smart-on-fhir`. Never commit to `main` directly, never force-push.
This phase touches every existing auth path in the app — be conservative, keep the dual-auth
window described below, and don't remove the old session auth path until it's explicitly safe
to.

## 6A — OAuth2 authorisation server

| Item | Acceptance criteria |
|------|---------------------|
| P6-01 | Deploy an OIDC-compliant authorisation server — Keycloak (self-hosted), added to `docker-compose.yml`. Configure realms for staff and patient portal. |
| P6-02 | Migrate Flask session auth to OAuth2 Bearer token validation. Flask routes check JWT access tokens from Keycloak. **Maintain dual auth (session + Bearer) for a minimum 4-week window — do not cut over the old auth path in this same change.** |
| P6-03 | Implement SMART on FHIR launch sequence: EHR launch (`iss` parameter), standalone launch, patient context selection. FHIR SMART configuration endpoint at `/.well-known/smart-configuration`. |
| P6-04 | Implement SMART scopes: `patient/*.read`, `user/*.read`, `offline_access`, `launch/patient`, `launch/encounter`. Enforce scope checks on every FHIR resource request, not just the entry point. |
| P6-05 | Add FHIR R4 `CapabilityStatement` with `security.service = "SMART-on-FHIR"` and the SMART capabilities extension array. |

## 6B — LDAP / Active Directory federation

| Item | Acceptance criteria |
|------|---------------------|
| P6-06 | Configure Keycloak LDAP user federation to the hospital AD/OpenLDAP. Map AD groups to HMIS roles (doctor, nurse, pharmacist, admin, lab-tech, radiology). Test against a read-only service account before enabling write-back. |
| P6-07 | Implement SCIM 2.0 provisioning endpoint for automated user creation/deactivation from the HR system. Map to the existing `User` model. |
| P6-08 | Configure Keycloak session revocation: deactivating a user in AD propagates to Keycloak within 60 seconds and invalidates all active sessions. |

## 6C — Patient portal SSO

| Item | Acceptance criteria |
|------|---------------------|
| P6-09 | Migrate `PatientUser` authentication to Keycloak with a separate realm. Support username/password, TOTP MFA, optional social login (Google/Apple). |
| P6-10 | Implement patient-facing SMART standalone launch: patient logs in, consents to data scope, receives an access token scoped to their own records only. |

## Risks to handle explicitly

- Keycloak needs its own PostgreSQL database, HA deployment, and monitoring — size it as
  infrastructure, not a drop-in library.
- Run dual auth for the full 4-week minimum, monitor error rates, and only deprecate
  session auth after two consecutive weeks of zero errors on the new path. Do not remove the
  old path as part of this same change even if it appears to work.
- Test AD attribute mapping explicitly against a read-only service account before enabling
  any write-back to the directory.

## Done when

- Full test suite passes (including existing auth-dependent tests — check these carefully,
  this phase is the highest-risk one for silently breaking existing login flows); ruff clean.
- Dual auth is confirmed working for both the old and new path, not just the new one.
- Branch pushed (not force-pushed), ready for PR, with an explicit note on when the old
  session-auth path is safe to remove (not removed in this PR).
