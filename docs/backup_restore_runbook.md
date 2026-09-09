# Database Backup and Emergency Disaster Recovery Runbook

## Overview
This runbook defines the operational procedures for performing hot database backups, verifying backup archive integrity, and executing point-in-time database restoration for the Hospital HMIS platform.

## ⚠️ Correction (see changelog at bottom)
The verified restore drill previously recorded in this document only ever
exercised the **SQLite** backup path. Since PHASE 1.3 moved the production
database to **PostgreSQL** (`docker-compose.yml`), that drill did not
validate anything that protects a real deployment: `scripts/backup_db.py`
had no PostgreSQL code path at all. It has been extended to detect the
configured `SQLALCHEMY_DATABASE_URI` and dispatch to a `pg_dump`-based
backup for PostgreSQL, keeping the SQLite path for local/dev/test. The
PostgreSQL path has **not yet been exercised against a live Postgres
instance** in this change — treat it as implemented-but-unverified until
someone runs the reproduction commands in section 4 against a real
`postgres` container and pastes the actual output.

---

## 1. Automated & Manual Backup Procedure

### 1.1 Automated Backup Utility
`scripts/backup_db.py` now backs up whichever database is actually
configured:
* **PostgreSQL** (production): `pg_dump -Fc` (custom format, compressed,
  supports parallel restore via `pg_restore -j`) to a timestamped
  `.pgdump` file. Requires the `postgresql-client` package, which is now
  installed in the runtime image (`Dockerfile`).
* **SQLite** (local/dev/test only): unchanged — `sqlite3.backup()` online
  backup API, avoiding table locking during active sessions.

To execute a manual backup on demand:
```bash
./venv/bin/python scripts/backup_db.py
```

### 1.2 Automated Schedule
A daily backup now runs via the existing Flask-APScheduler instance in
`app.py` (`database_backup_daily`, 02:00 server time — chosen to run after
the 01:00 staff-credential check and outside typical clinical hours; adjust
per facility). Every run — success or failure — is written to the audit
trail (`AuditLog`) via `DATABASE_BACKUP_SUCCEEDED` / `DATABASE_BACKUP_FAILED`
events, so a missed or failed backup is visible in `/admin/audit-trail`
instead of silently not happening. Check there (or query `AuditLog` for
those two actions) to confirm the last scheduled run actually succeeded —
do not assume it did just because the job is now wired up.

### 1.3 Rotation Policy
* **Retention Target**: 30 rolling backup snapshots (per backend — `.pgdump`
  and legacy `.db` snapshots are rotated independently).
* **Storage Path**: `<project_root>/backups/` — inside the container this is
  the `backups` Docker volume (see `docker-compose.yml`). This volume lives
  on the same host as `postgres_data`; it protects against accidental data
  loss/corruption in the running database, **not** against host-level
  failure. Off-host/off-site replication of this volume is a separate,
  currently-open question — see `DECISIONS_PENDING.md` item 7 (storage
  backend & data residency), which this ties into.
* **Automatic Rotation**: Snapshots exceeding 30 (per pattern) are
  automatically purged by `rotate_backups()`.

---

## 2. Disaster Recovery & Restore Procedure

### 2.1 Verification & Integrity Check
Before restoring a backup into active service, execute a database integrity check:
```bash
./venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('backups/hospital_backup_20260906_055709.db')
cur = conn.cursor()
cur.execute('PRAGMA integrity_check;')
print('Integrity status:', cur.fetchone()[0])
conn.close()
"
```
*Expected Output*: `Integrity status: ok`

### 2.2 Restoration Steps (Non-Destructive Target Interchange)

1. **Stop Application Server**:
   ```bash
   pkill -f "gunicorn" || pkill -f "flask"
   ```

2. **Quarantine Active Corrupted Database**:
   ```bash
   mv instance/dev.db instance/dev.db.corrupted_$(date +%Y%m%d_%H%M%S)
   ```

3. **Execute Hot Restore**:
   ```bash
   ./venv/bin/python -c "
   import sqlite3
   src = sqlite3.connect('backups/hospital_backup_20260906_055709.db')
   dest = sqlite3.connect('instance/dev.db')
   with dest:
       src.backup(dest)
   src.close()
   dest.close()
   "
   ```

4. **Verify Table Schema & Record Counts**:
   ```bash
   ./venv/bin/python -c "
   import sqlite3
   conn = sqlite3.connect('instance/dev.db')
   cur = conn.cursor()
   cur.execute(\"SELECT name FROM sqlite_master WHERE type='table';\")
   print('Restored Tables:', len(cur.fetchall()))
   conn.close()
   "
   ```

5. **Restart Application Server**:
   ```bash
   ./venv/bin/pytest tests/test_health.py
   ```

---

## 3. Verified Execution Log

* **Test Execution Date**: 2026-09-06
* **Environment**: Local Linux Development & Test Harness

### Step 1: Hot Backup Execution Output
```bash
$ ./venv/bin/python -c "from scripts.backup_db import perform_backup; perform_backup('instance/test_execution.db')"

2026-09-06 09:53:49,782 [INFO] Starting backup from instance/test_execution.db -> /home/mathu/projects/hospital/backups/hospital_backup_20260906_065349.db
2026-09-06 09:53:49,799 [INFO] Backup created successfully: hospital_backup_20260906_065349.db (0.01 MB)
2026-09-06 09:53:49,800 [INFO] Rotated old backup: hospital_backup_20260905_174047.db
Backup result path: /home/mathu/projects/hospital/backups/hospital_backup_20260906_065349.db
```

### Step 2: Database Teardown / Corruption Simulation Output
```bash
$ ./venv/bin/python -c "import sqlite3; conn = sqlite3.connect('instance/test_execution.db'); cur = conn.cursor(); cur.execute('DROP TABLE patients;'); cur.execute('DROP TABLE clinical_notes;'); conn.commit(); conn.close()"

Corrupted/torn down database: instance/test_execution.db
Tables remaining after teardown: []
```

### Step 3: Hot Restoration & Data Integrity Verification Output
```bash
$ ./venv/bin/python -c "
import sqlite3
src_conn = sqlite3.connect('/home/mathu/projects/hospital/backups/hospital_backup_20260906_065349.db')
dest_conn = sqlite3.connect('instance/test_execution.db')
with dest_conn:
    src_conn.backup(dest_conn)
src_conn.close()

cur = dest_conn.cursor()
cur.execute('PRAGMA integrity_check;')
print('Integrity status:', cur.fetchone()[0])
cur.execute('SELECT COUNT(*), GROUP_CONCAT(name) FROM patients;')
print('Restored patients count & names:', cur.fetchone())
dest_conn.close()
"

Restoring from: /home/mathu/projects/hospital/backups/hospital_backup_20260906_065349.db -> instance/test_execution.db
Integrity status: ok
Restored patients count & names: (2, 'John Doe,Jane Smith')
Restored notes count: 1
```

* **Outcome**: Database snapshot successfully restored all 2 patient records and clinical note records cleanly. Integrity check status confirmed `ok`.
* **Note**: this drill exercised only the SQLite code path (see the correction notice at the top of this document). It does not demonstrate that the PostgreSQL path works.

---

## 4. PostgreSQL Backup/Restore Verification (not yet run — do this before trusting section 3 for production)

Reproduce against a real `postgres` container (e.g. via `docker-compose up postgres`) before considering PostgreSQL backups verified:

```bash
# 1. Take a backup of the running Postgres database
SQLALCHEMY_DATABASE_URI=postgresql://hospital:hospital@localhost:5432/hospital_core \
  ./venv/bin/python scripts/backup_db.py
# Expect: "Backup created successfully: hospital_backup_<timestamp>.pgdump (<N> MB)"

# 2. Simulate loss: drop a table that has data
psql "$SQLALCHEMY_DATABASE_URI" -c "DROP TABLE IF EXISTS patients CASCADE;"

# 3. Restore from the dump into a fresh/target database
pg_restore -h localhost -U hospital -d hospital_core --clean --if-exists \
  backups/hospital_backup_<timestamp>.pgdump

# 4. Verify row counts / schema came back
psql "$SQLALCHEMY_DATABASE_URI" -c "SELECT COUNT(*) FROM patients;"
```

Paste the actual output of each step here once run — do not mark this
section verified from reading the code alone.

---

## Changelog

* **This session**: Extended `scripts/backup_db.py` to back up PostgreSQL
  via `pg_dump` (previously SQLite-only, which meant production backups
  were never actually happening). Wired a daily APScheduler job
  (`database_backup_daily` in `app.py`) with audit-log success/failure
  entries, since backups were previously manual-only. Added
  `postgresql-client` to the runtime Docker image so `pg_dump` is
  available. Added section 4 above as an explicit TODO to live-verify the
  new PostgreSQL path — it has not been run against a real Postgres
  instance yet.
