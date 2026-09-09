#!/usr/bin/env bash
# scripts/backup_postgres.sh
# ───────────────────────────
# Automated PostgreSQL backup for HMIS production database.
#
# Produces a timestamped compressed pg_dump, rotates to MAX_BACKUPS,
# and writes a per-backup manifest (row counts) for restore verification.
#
# Usage:
#   ./scripts/backup_postgres.sh             # backup production DB
#   ./scripts/backup_postgres.sh --verify <file.dump>  # restore-drill a backup
#
# Environment variables (override via .env or shell):
#   POSTGRES_HOST     (default: localhost)
#   POSTGRES_PORT     (default: 5432)
#   POSTGRES_DB       (default: hospital_core)
#   POSTGRES_USER     (default: hospital)
#   PGPASSWORD        must be set (via env or .pgpass) — never hardcode here
#   BACKUP_DIR        (default: ./backups/postgres)
#   MAX_BACKUPS       (default: 30)
#
# Pre-requisites: pg_dump, psql, gzip, python3 all in PATH.
# Run as the OS user that has read access to PostgreSQL.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env if present (but do not export blindly — only override unset vars)
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -o allexport
    # shellcheck disable=SC1090
    source "$REPO_ROOT/.env"
    set +o allexport
fi

PG_HOST="${POSTGRES_HOST:-localhost}"
PG_PORT="${POSTGRES_PORT:-5432}"
PG_DB="${POSTGRES_DB:-hospital_core}"
PG_USER="${POSTGRES_USER:-hospital}"
BACKUP_DIR="${BACKUP_DIR:-$REPO_ROOT/backups/postgres}"
MAX_BACKUPS="${MAX_BACKUPS:-30}"

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
DUMP_FILE="$BACKUP_DIR/hospital_pg_${TIMESTAMP}.dump.gz"
MANIFEST_FILE="$BACKUP_DIR/hospital_pg_${TIMESTAMP}.manifest.json"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }
fail() { log "ERROR: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command '$1' not found in PATH."
}

# ── Mode: restore-drill (--verify <file>) ────────────────────────────────────
if [[ "${1:-}" == "--verify" ]]; then
    DUMP="${2:-}"
    [[ -f "$DUMP" ]] || fail "Dump file not found: $DUMP"

    SCRATCH_DB="hmis_restore_drill_$$"
    log "Starting restore drill: $DUMP → scratch DB '$SCRATCH_DB'"

    require_cmd psql
    require_cmd pg_restore

    psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d postgres \
        -c "CREATE DATABASE $SCRATCH_DB;" \
        || fail "Could not create scratch DB $SCRATCH_DB"

    cleanup_scratch() {
        psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d postgres \
            -c "DROP DATABASE IF EXISTS $SCRATCH_DB;" 2>/dev/null || true
    }
    trap cleanup_scratch EXIT

    log "Restoring dump into $SCRATCH_DB ..."
    gunzip -c "$DUMP" | pg_restore \
        -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" \
        -d "$SCRATCH_DB" --no-owner --no-acl 2>&1 \
        || fail "pg_restore exited with errors — see output above"

    log "Verifying row counts ..."
    TABLES=$(psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$SCRATCH_DB" \
        -t -c "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename;")
    TOTAL_ROWS=0
    while IFS= read -r table; do
        table=$(echo "$table" | xargs)  # trim whitespace
        [[ -z "$table" ]] && continue
        COUNT=$(psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$SCRATCH_DB" \
            -t -c "SELECT COUNT(*) FROM \"$table\";" | xargs)
        log "  $table: $COUNT rows"
        TOTAL_ROWS=$((TOTAL_ROWS + COUNT))
    done <<< "$TABLES"

    log "Restore drill complete. Total rows verified: $TOTAL_ROWS"

    # Compare against manifest if available
    MANIFEST="${DUMP%.dump.gz}.manifest.json"
    if [[ -f "$MANIFEST" ]]; then
        EXPECTED=$(python3 -c "import json,sys; d=json.load(open('$MANIFEST')); print(d.get('total_rows', 'unknown'))")
        log "Manifest expected total_rows: $EXPECTED  |  Actual: $TOTAL_ROWS"
        if [[ "$EXPECTED" != "unknown" && "$EXPECTED" != "$TOTAL_ROWS" ]]; then
            fail "Row count mismatch: expected $EXPECTED, got $TOTAL_ROWS"
        fi
    fi

    log "✅ Restore drill PASSED"
    exit 0
fi

# ── Mode: backup ─────────────────────────────────────────────────────────────
require_cmd pg_dump
require_cmd gzip
require_cmd python3

mkdir -p "$BACKUP_DIR"

# Confirm PGPASSWORD or .pgpass is set — never accept a missing credential
if [[ -z "${PGPASSWORD:-}" ]] && [[ ! -f "${HOME}/.pgpass" ]]; then
    fail "PGPASSWORD env var is not set and ~/.pgpass does not exist. " \
         "Set PGPASSWORD or configure a .pgpass file before running backups."
fi

log "Starting PostgreSQL backup: $PG_DB@$PG_HOST:$PG_PORT → $DUMP_FILE"

# pg_dump with custom format (supports parallel restore), piped through gzip
pg_dump \
    -h "$PG_HOST" \
    -p "$PG_PORT" \
    -U "$PG_USER" \
    -d "$PG_DB" \
    --format=custom \
    --compress=0 \
    | gzip -9 > "$DUMP_FILE" \
    || fail "pg_dump failed"

SIZE_MB=$(du -m "$DUMP_FILE" | cut -f1)
log "Dump complete: $(basename "$DUMP_FILE") (${SIZE_MB} MB)"

# Write manifest (row counts snapshot for restore verification)
log "Writing manifest ..."
python3 - "$PG_HOST" "$PG_PORT" "$PG_DB" "$PG_USER" "$MANIFEST_FILE" << 'PYEOF'
import json, os, subprocess, sys
host, port, db, user, out = sys.argv[1:]
env = {**os.environ, "PGPASSWORD": os.environ.get("PGPASSWORD", "")}
result = subprocess.run(
    ["psql", "-h", host, "-p", port, "-U", user, "-d", db,
     "-t", "-c",
     "SELECT json_object_agg(tablename, cnt) FROM "
     "(SELECT tablename, (xpath('/row/cnt/text()', query_to_xml("
     "'SELECT COUNT(*) AS cnt FROM \"'||tablename||'\"',false,true,'')))[1]::text::int AS cnt "
     "FROM pg_tables WHERE schemaname='public') sub;"],
    capture_output=True, text=True, env=env
)
table_counts = {}
total = 0
try:
    table_counts = json.loads(result.stdout.strip()) or {}
    total = sum(table_counts.values())
except Exception:
    pass
manifest = {"db": db, "host": host, "table_counts": table_counts, "total_rows": total}
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"  Manifest written: {total} rows across {len(table_counts)} tables")
PYEOF

# Rotate old backups (keep MAX_BACKUPS most recent .dump.gz files)
log "Rotating backups (keeping $MAX_BACKUPS most recent) ..."
BACKUP_LIST=("$BACKUP_DIR"/hospital_pg_*.dump.gz)
COUNT=${#BACKUP_LIST[@]}
if (( COUNT > MAX_BACKUPS )); then
    EXCESS=$(( COUNT - MAX_BACKUPS ))
    for (( i=0; i<EXCESS; i++ )); do
        OLD="${BACKUP_LIST[$i]}"
        rm -f "$OLD" "${OLD%.dump.gz}.manifest.json" 2>/dev/null || true
        log "  Rotated: $(basename "$OLD")"
    done
fi

log "✅ Backup complete: $(basename "$DUMP_FILE")"
log "   Restore drill:  ./scripts/backup_postgres.sh --verify $DUMP_FILE"
