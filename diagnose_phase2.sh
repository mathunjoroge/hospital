#!/bin/bash
echo "=== 1. Establishing Test Baseline ==="
pytest -q --cov --tb=no | tail -n 5

echo -e "\n=== 2. Verifying P2-11: HSTS Header (Already Shipped?) ==="
grep -n "Strict-Transport-Security" app.py || echo "❌ HSTS header NOT found in app.py"

echo -e "\n=== 3. Verifying Automated DB Backups (Already Shipped?) ==="
if [ -f scripts/backup_db.py ]; then
    echo "✅ scripts/backup_db.py exists."
    head -n 15 scripts/backup_db.py
else
    echo "❌ scripts/backup_db.py NOT found."
fi

echo -e "\n=== 4. Checking Current Sentry/OTel State in app.py ==="
grep -in -C 2 "sentry\|otel\|opentelemetry\|tracing" app.py || echo "No OTel/Sentry config found in app.py."

echo -e "\n=== 5. Checking DECISIONS_PENDING.md Item 3 ==="
grep -A 10 "## 3. Telemetry" DECISIONS_PENDING.md
