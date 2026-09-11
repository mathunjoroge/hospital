#!/bin/bash
echo "=== 1. Current OAuth / Auth libraries ==="
grep -i "oauth\|authlib\|jwt\|oidc" requirements.txt

echo -e "\n=== 2. Current FHIR endpoints ==="
grep -n "@fhir_bp.route" departments/api/fhir.py

echo -e "\n=== 3. Existing SSO/SMART decisions ==="
grep -i -C 2 "SSO\|SMART\|OAuth\|OIDC" DECISIONS_PENDING.md || echo "No SSO decisions recorded yet."

echo -e "\n=== 4. Current User model capabilities ==="
grep -n "class User\|facility_id\|roles\|scopes" departments/models/user.py | head -n 15

echo -e "\n=== 5. Check for existing OAuth/SMART routes ==="
find departments -name "*oauth*" -o -name "*smart*" -o -name "*launch*" | grep -v __pycache__

echo -e "\n=== 6. Check config.py for OAuth/JWT settings ==="
grep -i "oauth\|jwt\|secret_key\|smart" departments/config.py | head -n 15

echo -e "\n=== 7. Check extensions.py for auth setup ==="
grep -n "jwt\|login_manager\|oauth\|JWTManager" extensions.py
