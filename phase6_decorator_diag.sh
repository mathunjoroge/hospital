#!/bin/bash
echo "=== 1. Where is jwt_or_session_required defined? ==="
grep -rn "def jwt_or_session_required\|jwt_or_session_required =" departments/ app.py extensions.py | grep -v ".pyc"

echo -e "\n=== 2. Where is roles_required defined? ==="
grep -rn "def roles_required\|roles_required =" departments/ app.py extensions.py | grep -v ".pyc"

echo -e "\n=== 3. Full auth.py content (JWT login endpoint) ==="
cat departments/api/auth.py

echo -e "\n=== 4. Check for existing OAuth2 models in database ==="
grep -rn "class.*OAuth\|oauth_client\|oauth_token" departments/models/ | grep -v ".pyc"

echo -e "\n=== 5. How is roles_required used in the codebase? ==="
grep -rn "@roles_required" departments/ | head -n 10
