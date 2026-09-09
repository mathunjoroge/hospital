# Database Backup and Emergency Disaster Recovery Runbook

## Overview

This runbook defines the operational procedures for backing up, verifying,
and restoring the HMIS production database.

**Production database: PostgreSQL** (via Docker or managed service).  
Testing/CI uses SQLite in-memory only — the SQLite backup utility in
`scripts/backup_db.py` applies to CI only and is **not** a production
backup mechanism.

---

## 1. Production Backup — PostgreSQL (`scripts/backup_postgres.sh`)

### 1.1 Prerequisites

```bash
# Required system tools
pg_dump --version   # PostgreSQL client tools
psql --version
gzip --version
python3 --version

# Required environment variables (set in .env or shell)
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=hospital_core
export POSTGRES_USER=hospital
export PGPASSWORD=<your_db_password>   # or configure ~/.pgpass
```

Never hardcode `PGPASSWORD` in scripts or commit it to git.
Use `.env` (gitignored) or a secrets manager in production.

### 1.2 Run a Manual Backup

```bash
./scripts/backup_postgres.sh
```

Output (example):
```
[2026-09-09T12:00:00Z] Starting PostgreSQL backup: hospital_core@localhost:5432
[2026-09-09T12:00:02Z] Dump complete: hospital_pg_20260909_120000.dump.gz (4 MB)
[2026-09-09T12:00:02Z] Manifest written: 187,341 rows across 63 tables
[2026-09-09T12:00:02Z] ✅ Backup complete: hospital_pg_20260909_120000.dump.gz
```

### 1.3 Backup Location and Rotation

| Item | Value |
|---|---|
| Default directory | `./backups/postgres/` |
| Naming | `hospital_pg_YYYYMMDD_HHMMSS.dump.gz` |
| Rotation | 30 most recent kept; older deleted automatically |
| Manifest | `hospital_pg_YYYYMMDD_HHMMSS.manifest.json` (row counts per table) |

### 1.4 Automate with Cron (Daily at 02:00 UTC)

```cron
0 2 * * * /path/to/hospital/scripts/backup_postgres.sh >> /var/log/hmis_backup.log 2>&1
```

---

## 2. Restore Drill — Verify a Backup

The `--verify` flag restores a dump into a throwaway database, counts rows,
and compares against the manifest. Run this monthly or before every major
deployment.

```bash
./scripts/backup_postgres.sh --verify backups/postgres/hospital_pg_20260909_120000.dump.gz
```

Expected output:
```
[2026-09-09T...] Starting restore drill → scratch DB 'hmis_restore_drill_12345'
[2026-09-09T...] Restoring dump ...
[2026-09-09T...] Verifying row counts ...
[2026-09-09T...]   patients: 3482 rows
[2026-09-09T...]   ...
[2026-09-09T...] Restore drill complete. Total rows verified: 187341
[2026-09-09T...] Manifest expected total_rows: 187341 | Actual: 187341
[2026-09-09T...] ✅ Restore drill PASSED
```

The script drops the scratch DB automatically on exit (success or failure).

---

## 3. Production Restore Procedure

Only invoke this after a restore drill has confirmed the target backup is
sound. Coordinate with the clinical team — there will be brief downtime.

```bash
# 1. Stop application server
pkill -f gunicorn || true

# 2. Create a named restore target (do NOT overwrite production directly first)
RESTORE_DB="hospital_core_restore_$(date +%Y%m%d_%H%M%S)"
psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d postgres \
    -c "CREATE DATABASE $RESTORE_DB;"

# 3. Restore the dump
gunzip -c backups/postgres/hospital_pg_YYYYMMDD_HHMMSS.dump.gz \
  | pg_restore -h "$POSTGRES_HOST" -U "$POSTGRES_USER" \
    -d "$RESTORE_DB" --no-owner --no-acl

# 4. Sanity-check row counts in the restore target
psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$RESTORE_DB" \
    -c "SELECT COUNT(*) FROM patients;"

# 5. Once verified: swap databases (rename requires no active connections)
psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d postgres << SQL
ALTER DATABASE hospital_core RENAME TO hospital_core_pre_restore_$(date +%Y%m%d);
ALTER DATABASE $RESTORE_DB RENAME TO hospital_core;
SQL

# 6. Restart application server
gunicorn app:app -w 4 -b 0.0.0.0:8000

# 7. Run health check
curl http://localhost:8000/healthz
```

Keep `hospital_core_pre_restore_YYYYMMDD` for at least 72 hours before
dropping it, in case the restore exposed a data issue.

---

## 4. RPO / RTO Targets

| Target | Value | Notes |
|---|---|---|
| RPO (data loss tolerance) | < 24 hours | Daily cron backup. Reduce to < 1 hour by enabling PostgreSQL WAL archiving to S3/MinIO. |
| RTO (recovery time) | < 2 hours | Includes restore + drill + health check. Reduce by keeping a warm standby via `pg_basebackup`. |

WAL archiving configuration is a DECISIONS_PENDING item (storage backend
and data residency — see `DECISIONS_PENDING.md` item 7).

---

## 5. CI / Testing Backup (SQLite Only)

The legacy `scripts/backup_db.py` utility creates SQLite backups and is
used in CI environments only:

```bash
python3 scripts/backup_db.py
```

This is **not** a substitute for the PostgreSQL backup in production.
