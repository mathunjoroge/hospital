# Database Backup and Emergency Disaster Recovery Runbook

## Overview
This runbook defines the operational procedures for performing hot database backups, verifying backup archive integrity, and executing point-in-time database restoration for the Hospital HMIS platform.

---

## 1. Automated & Manual Backup Procedure

### 1.1 Automated Backup Utility
The backup engine (`scripts/backup_db.py`) creates timestamped hot backups using SQLite online backup API (`sqlite3.backup`), avoiding database table locking during active clinical sessions.

To execute a manual backup on demand:
```bash
./venv/bin/python scripts/backup_db.py
```

### 1.2 Rotation Policy
* **Retention Target**: 30 rolling backup snapshots.
* **Storage Path**: `/home/mathu/projects/hospital/backups/hospital_backup_YYYYMMDD_HHMMSS.db`.
* **Automatic Rotation**: Snapshots exceeding 30 are automatically purged by `rotate_backups()`.

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

## 3. Empirical Verification Record

* **Execution Date**: 2026-09-06
* **Backup Created**: `backups/hospital_backup_20260906_055709.db` (Size: 0.01 MB)
* **Restore Validation**: Restored cleanly to temporary instance `test_restored.db`.
* **PRAGMA Integrity Result**: `ok`
