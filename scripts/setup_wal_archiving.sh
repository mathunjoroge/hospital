#!/usr/bin/env bash
# scripts/setup_wal_archiving.sh
# ─────────────────────────────────────────────────────────────────────────────
# P2-07: Continuous WAL Archiving — setup helper.
#
# This script:
#   1. Explains the required postgresql.conf changes (cannot be applied from
#      here — a DBA must edit the running Postgres container config).
#   2. Starts pg_receivewal to stream WAL segments to a local backup directory.
#
# Prerequisites:
#   • POSTGRES_PASSWORD, POSTGRES_USER, POSTGRES_DB, POSTGRES_HOST must be
#     set in the environment (source .env first).
#   • postgresql-client-16 must be installed on the host running this script.
#   • The Postgres server must have wal_level=replica and max_wal_senders >= 2.
#
# After running this script, test PITR recovery on a staging DB before
# declaring the RPO=4h SLA met. See docs/dr_runbook.md for the drill procedure.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

: "${POSTGRES_HOST:?POSTGRES_HOST must be set}"
: "${POSTGRES_USER:?POSTGRES_USER must be set}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}"
: "${POSTGRES_DB:?POSTGRES_DB must be set}"
: "${POSTGRES_PORT:=5432}"

WAL_ARCHIVE_DIR="${WAL_ARCHIVE_DIR:-$(pwd)/backups/wal_archive}"
mkdir -p "$WAL_ARCHIVE_DIR"

cat <<'INSTRUCTIONS'
═══════════════════════════════════════════════════════════════════════════════
REQUIRED: postgresql.conf changes (apply inside the postgres container)
═══════════════════════════════════════════════════════════════════════════════
Run:
  docker compose exec postgres psql -U "$POSTGRES_USER" -c \
    "ALTER SYSTEM SET wal_level = 'replica';"
  docker compose exec postgres psql -U "$POSTGRES_USER" -c \
    "ALTER SYSTEM SET max_wal_senders = 3;"
  docker compose exec postgres psql -U "$POSTGRES_USER" -c \
    "ALTER SYSTEM SET wal_keep_size = '512MB';"
  docker compose exec postgres psql -U "$POSTGRES_USER" -c \
    "SELECT pg_reload_conf();"

Then create the replication slot (idempotent):
  docker compose exec postgres psql -U "$POSTGRES_USER" -c \
    "SELECT pg_create_physical_replication_slot('hmis_wal_receiver', true)
     WHERE NOT EXISTS (
       SELECT 1 FROM pg_replication_slots
       WHERE slot_name = 'hmis_wal_receiver'
     );"

A PostgreSQL restart may be required if wal_level was previously set to
'minimal'. Check with: SHOW wal_level;
═══════════════════════════════════════════════════════════════════════════════
INSTRUCTIONS

echo "[setup_wal_archiving] WAL archive directory: $WAL_ARCHIVE_DIR"
echo "[setup_wal_archiving] Starting pg_receivewal — streaming WAL to $WAL_ARCHIVE_DIR"
echo "[setup_wal_archiving] Press Ctrl-C to stop. Run this under a process supervisor in production."
echo ""

export PGPASSWORD="$POSTGRES_PASSWORD"
exec pg_receivewal \
  --host="$POSTGRES_HOST" \
  --port="$POSTGRES_PORT" \
  --username="$POSTGRES_USER" \
  --slot="hmis_wal_receiver" \
  --directory="$WAL_ARCHIVE_DIR" \
  --no-password \
  --verbose \
  --synchronous
