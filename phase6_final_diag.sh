#!/bin/bash
echo "=== 1. FULL rbac.py (how roles_required resolves the user) ==="
cat departments/rbac.py

echo -e "\n=== 2. EVERY usage of jwt_or_session_required repo-wide ==="
grep -rn "jwt_or_session_required" departments/ app.py tests/ | grep -v ".pyc"

echo -e "\n=== 3. EVERY usage of roles_required repo-wide ==="
grep -rn "@roles_required\|roles_required(" departments/ app.py tests/ | grep -v ".pyc" | wc -l
grep -rn "@roles_required" departments/ | grep -v ".pyc" | awk -F: '{print $1}' | sort | uniq -c

echo -e "\n=== 4. departments/api/__init__.py (blueprint definition) ==="
cat departments/api/__init__.py

echo -e "\n=== 5. Full User model (all columns) ==="
sed -n '1,60p' departments/models/user.py

echo -e "\n=== 6. Confirm Section 20 NOT yet in DECISIONS_PENDING.md ==="
grep -c "Section 20" DECISIONS_PENDING.md
