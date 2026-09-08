# Claude Code Prompt: HIMS World-Class / Tertiary-Hospital Readiness Remediation

Paste this directly into Claude Code at the repo root (`mathunjoroge/hospital`).

---

## Context (read first, do not skip)

You are working on a Flask-based Health Information Management System (HIMS)
currently at "strong national-grade prototype" maturity per its own
`docs/worldclass/ROADMAP.md`. An external audit (human + AI, code actually
executed and tests actually run — not just read) found the following
**confirmed, reproduced defects**, plus the pre-existing gap list the repo
already tracks honestly in `DECISIONS_PENDING.md` and `docs/worldclass/ROADMAP.md`.

Do not treat this as a greenfield rewrite. Treat it as a hardening pass on a
codebase whose author has already been disciplined about flagging unknowns.
Preserve that discipline: **do not silently make clinical, legal, financial,
or architectural decisions that the repo has already flagged as
`HARD STOP` / pending human sign-off in `DECISIONS_PENDING.md`.** Where your
fix touches one of those items, stop and add a note to
`DECISIONS_PENDING.md` instead of guessing.

Work in small, reviewable commits, one fix category at a time. After each
category, run the relevant test subset and paste the actual passing output
before moving to the next category — do not claim a fix is done from reading
the diff alone.

---

## Priority 0 — Patient-safety-critical defect (fix first, alone, in its own PR)

**File:** `departments/medicine/cdss.py` + `departments/shared/drugcentral.py`

**Confirmed bug:** `check_drug_interactions()` calls
`query_drugcentral_ddi()`, which opens a `psycopg2.connect()` to an external
public host (`unmtid-dbs.net:5433`) **with no `connect_timeout` set**. When
that host is unreachable — any hospital firewall, air-gapped deployment, or
outage of that one external server — the call hangs indefinitely rather than
failing fast. The local `KNOWN_INTERACTIONS` fallback matrix only executes
*after* the live query returns, so the safety net never engages if the
network call hangs rather than errors. This means the primary prescribing
safety check for drug-drug interactions can silently stall a clinical
workflow with no fallback and no operator-visible error.

**Fix requirements:**
1. Add `connect_timeout` (2–3 seconds, make it configurable via env var) to
   `DRUGCENTRAL_DB_PARAMS` in `departments/shared/drugcentral.py`.
2. Wrap the entire live-lookup path in `query_drugcentral_ddi()` in a
   try/except that catches `psycopg2.OperationalError`, `psycopg2.Error`, and
   generic `Exception`, logs a structured warning (include latency and
   failure reason), and returns `[]` immediately so
   `check_drug_interactions()` always falls through to the local matrix.
3. Add a circuit breaker: if the live DrugCentral lookup has failed N times
   in a row (config value, default 3), skip attempting it for a cooldown
   window (default 5 minutes) and go straight to local fallback. Log when the
   breaker opens and closes.
4. This same "external network dependency with no timeout" pattern may exist
   elsewhere — grep the codebase for every `psycopg2.connect`, `requests.get`,
   `requests.post`, and any other outbound network client construction, and
   confirm each one has an explicit timeout and a defined fallback/error path.
   List every instance you find and its fix in the PR description, even ones
   you don't think are urgent — flag severity per instance.
5. Add a regression test that simulates an unreachable DrugCentral host
   (mock/patch the connection to raise or hang) and asserts
   `check_drug_interactions()` returns local-matrix results within a bounded
   time (e.g. under 1 second in test).
6. Do the same audit-and-fix pass for any other clinical decision support
   path that depends on an external service (allergy checking, dosing
   guidance) — confirm they either have no live external dependency, or have
   the same timeout + fallback + circuit breaker treatment.

Do not proceed to anything below until this PR is merged and its new
regression test is green.

---

## Priority 1 — Performance & concurrency (throughput regresses under load)

**Evidence:** `docs/load_test_results.md` shows throughput *falling* as
concurrency rises (42 → 16 req/s from 5 to 20 threads on `/healthz` alone),
and `POST /billing/pay_bills` p95 latency reaching 1.15s at 20 threads. This
pattern (throughput dropping, not plateauing) indicates serialization
somewhere in the request path, not honest capacity limits.

1. Re-run `scripts/load_test_baseline.py` against the app served by
   **gunicorn with multiple sync or eventlet workers** (per
   `requirements.txt`, both are already dependencies), not the Flask dev
   server, and against PostgreSQL, not SQLite. Confirm whether the
   regression persists — if it disappears, the original report was
   measuring the dev server, not the app, and the docs should say so
   explicitly instead of implying a production benchmark.
2. If the regression persists under gunicorn+Postgres, profile
   `/billing/pay_bills` and `/admin/analytics` specifically (the two
   worst offenders) for N+1 queries, missing indexes, or long-held locks.
   Use `EXPLAIN ANALYZE` on the actual queries generated by SQLAlchemy.
3. Confirm `SQLALCHEMY_ENGINE_OPTIONS` pool settings in `config.py`
   (`pool_size=20, max_overflow=30`) are actually sufficient for the worker
   count you land on, and that nothing is holding a connection open across
   a request (e.g. session not closed on early return/exception paths).
4. Rewrite `docs/load_test_results.md` with the corrected methodology and
   results, and be explicit in the doc about what environment (workers,
   DB engine, hardware) the numbers came from. Do not publish "concurrency
   baseline" numbers gathered from a single-process dev server again.

---

## Priority 2 — Test coverage and hardening

**Evidence:** CI enforces only `--cov-fail-under=20`; actual measured
coverage in a live run was ~34.5%. Zero-coverage modules include
`departments/security_ops/models.py`, `departments/system_ops/models.py`,
`departments/offline_sync/models.py`, `departments/public_health/models.py`.

1. Raise `--cov-fail-under` incrementally (20 → 40 → 60) as real coverage
   improves — do not jump the gate ahead of actual tests, and do not lower
   it to make CI pass.
2. Prioritize test-writing for: RBAC/`@roles_required` edge cases,
   break-glass expiry/justification handling, billing sync idempotency
   (`departments/billing/sync.py` — flagged as a known gap in
   `DECISIONS_PENDING.md` item 10 for update/delete handling), and the
   consent-gating logic (`has_ai_consent`).
3. For every module currently at 0% coverage, either add tests or add a
   comment explaining why it's intentionally untested (e.g. dead code
   scheduled for removal) — an unexplained 0% file is itself a finding.
4. Add a mutation-testing or fault-injection pass (e.g. `mutmut` or hand
   written) specifically for the billing and pharmacy dispensing modules,
   since silent financial or dosing errors are the highest-cost failure mode.

---

## Priority 3 — Operational maturity (per `docs/worldclass/ROADMAP.md` Phase 1–2)

Work through these in the order the roadmap already specifies. For each item,
implement the smallest version that is real and tested, not a stub:

1. **Observability**: structured logging with request IDs, a
   `/metrics` endpoint (Prometheus format) covering request latency, error
   rate, and queue depth (Celery), and integration with an error tracker.
   Per `DECISIONS_PENDING.md` item 3, **do not pick Sentry SaaS vs
   self-hosted yourself** — implement against an abstraction that can point
   at either, and leave the DECISIONS_PENDING item open for the human
   choice.
2. **Backups**: automate what `docs/backup_restore_runbook.md` currently
   documents as a manual runbook — a scheduled job, a restore drill script,
   and a test that actually restores a backup into a scratch DB and verifies
   row counts/checksums match.
3. **Security headers & session hardening**: confirm `Secure`, `HttpOnly`,
   `SameSite` are set correctly on the Flask-Session cookie in production
   config (not just test config), add CSP headers, and add a test that
   asserts these headers are present on a real response.
4. **Secrets**: confirm `SECRET_KEY`, `ENCRYPTION_KEY`, and DB credentials
   are never defaulted to a hardcoded value outside of `FLASK_ENV=testing`.
   Audit `config.py` line by line for this — it currently has a hardcoded
   default Postgres password (`postgresql://hospital:hospital@...`) as a
   fallback; confirm this path is unreachable outside test config and add a
   startup check that refuses to boot in production mode with a default
   secret.

---

## Priority 4 — Clinical/compliance items already flagged as HARD STOP

Do not implement these unilaterally. For each one in `DECISIONS_PENDING.md`,
your job is to:
- Prepare the two or three concrete implementation options,
- Note the trade-offs of each,
- Leave the decision itself for the human stakeholder,
- Only build the code path once an explicit answer is given.

This applies especially to: controlled drug register dual-signature policy,
KRA eTIMS integration, medical file storage residency, and the
single-facility vs multi-facility architecture decision — these have real
legal/regulatory consequences if guessed wrong.

---

## What "world-class" means for this repo specifically — acceptance criteria

Before calling any of this "Johns Hopkins-grade," the following must all be
true, with evidence (test output, benchmark output, doc links), not just
code:

1. No clinical safety function can hang or silently fail without a bounded
   timeout, a fallback, and an audit log entry.
2. Load test numbers are gathered against a production-like stack
   (gunicorn multi-worker + PostgreSQL) and show throughput scaling or
   plateauing under concurrency, not degrading.
3. Test coverage is >60% with named exceptions for the rest, and specifically
   covers every clinical-safety and financial code path.
4. Every `HARD STOP` item in `DECISIONS_PENDING.md` is either resolved with
   a documented human decision, or still open and explicitly still blocking
   production deployment (i.e., the doc is a living gate, not a graveyard).
5. A real third-party security audit (not `bandit`/`pip-audit` alone) has
   been scoped and, ideally, run — this codebase cannot self-certify its
   own penetration-test readiness.
6. `docs/worldclass/ROADMAP.md` is updated to move completed items from
   "Critical Missing Areas" into a "Delivered" section with evidence links,
   and the doc's own self-description at the top ("strong national-grade
   prototype") is only upgraded once the criteria above are actually met —
   not before.

Do not mark this roadmap "complete" or claim world-class status in any
README or doc edit until items 1–4 above are independently verifiable by
someone re-running the tests/benchmarks themselves.
