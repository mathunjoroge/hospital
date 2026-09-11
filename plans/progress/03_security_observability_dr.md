# Prompt 3 of 9 — Observability, Disaster Recovery & Security Hardening (Phase 2)

Priority: High. P2-14 (row-level security) is an explicit prerequisite for Phase 6 (SSO) —
finish this phase before starting `04_sso_smart_on_fhir.md`.

Branch: `feat/phase2-observability-dr-security`. Never commit to `main` directly, never
force-push. Run the full test suite before starting and record the baseline.

Two of the items below (HSTS header, automated DB backups) are already shipped — verify
first (`grep -n "Strict-Transport-Security" app.py`, check `scripts/backup_db.py`) rather than
redoing them; only work the remaining gap.

## 2A — Distributed tracing & error tracking

| Item | Acceptance criteria |
|------|---------------------|
| P2-01 | Select and deploy an OpenTelemetry collector (self-hosted). Decision between Jaeger and Grafana Tempo needs recording in `DECISIONS_PENDING.md` item 3 before implementation — if it's not resolved there, stop and ask rather than picking one. |
| P2-02 | Instrument the Flask app with `opentelemetry-instrumentation-flask`. Propagate trace context into Celery tasks. Emit spans for DB queries >50ms, external API calls, FHIR endpoints. |
| P2-03 | Deploy Sentry self-hosted (DPA 2019 requires data residency — do not use Sentry SaaS US/global region; self-hosted or Sentry EU region only). Configure DSN in all services. Alert threshold: >5 new errors/min → PagerDuty. Currently `app.py` only has an inert comment noting this decision is pending — that's the actual current state, not a stub to leave alone. |
| P2-04 | Build an uptime dashboard (Grafana or Metabase): request rate, p50/p95/p99 latency, error rate, DB connection pool, Celery queue depth, Redis memory. 30-day retention. |
| P2-05 | Document SLOs: API availability ≥99.5%, prescription sign-off p95 <500ms, lab result delivery <2min from instrument receipt. Wire SLO burn-rate alerts. |

## 2B — Disaster recovery

| Item | Acceptance criteria |
|------|---------------------|
| P2-06 | Document RPO=4h, RTO=1h as a signed SLA — this needs hospital management sign-off, not just a config value. Flag as needing human sign-off in the PR if not already recorded. |
| P2-07 | Continuous WAL archiving from PostgreSQL primary to S3/MinIO or local NAS. Verify point-in-time recovery to within 5 minutes. (Basic backup automation already exists — extend it to continuous WAL archiving with PITR verification, don't rebuild from scratch.) |
| P2-08 | Script automated DR failover: promote read replica to primary, reconfigure `DATABASE_URL`, restart Celery workers. Document a quarterly DR drill procedure. |
| P2-09 | Deploy a hot-standby PostgreSQL read replica in a separate availability zone/server. Streaming replication lag alert if lag >30 seconds. |
| P2-10 | Automate DICOM backup to object storage. Separate retention: imaging 10 years (Kenya Medical Records Act), transactional data 7 years. |

## 2C — Security hardening (pre-pen-test remediation)

| Item | Acceptance criteria |
|------|---------------------|
| P2-11 | TLS 1.2+ everywhere, HTTP→HTTPS redirect, HSTS header. **Already shipped — verify, don't redo.** |
| P2-12 | Rotate all secrets to a secrets manager (HashiCorp Vault or AWS Secrets Manager). Remove secrets from environment variables in `docker-compose.yml`. |
| P2-13 | Content Security Policy headers. Expand `test_xss.py` to cover all form inputs, complete the SQL injection test suite. |
| P2-14 | Enable PostgreSQL row-level security (RLS) for multi-tenant data isolation between facilities. **This is a hard prerequisite for Phase 6 — do not skip or defer it.** |
| P2-15 | Weekly automated dependency audit (pip-audit, npm audit) in CI. Block merge if CVSS ≥7.0 advisories are unresolved. |

## Risks to handle explicitly

- Do not deploy OTel/Sentry to any SaaS region outside data-residency requirements — flag and
  stop rather than guess if the region isn't clearly compliant.
- Test the DR drill on a staging environment first. Do not run a failover drill against
  production as part of this task.
- Roll secrets rotation out service-by-service with a rollback plan per service — don't
  rotate everything in one shot.

## Done when

- Full test suite passes; ruff clean.
- P2-01's collector choice and P2-06's SLA are either already resolved in
  `DECISIONS_PENDING.md` or explicitly flagged as pending human sign-off in the PR — not
  silently decided by the agent.
- Branch pushed (not force-pushed), ready for PR.
