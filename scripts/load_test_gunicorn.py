#!/usr/bin/env python3
"""
scripts/load_test_gunicorn.py
──────────────────────────────
Proper HIMS load test against a real HTTP server (gunicorn + Postgres).

Unlike load_test_baseline.py which uses Flask's test_client() (single-process,
GIL-bound, no real networking), this script fires real HTTP requests at a
running gunicorn instance so results reflect actual concurrency behaviour.

Usage:
    # Start gunicorn first:
    #   SQLALCHEMY_DATABASE_URI="postgresql://mathu@localhost/hospital_db" \\
    #   venv/bin/gunicorn app:app --workers 4 --bind 127.0.0.1:8765
    #
    # Then run this script:
    #   venv/bin/python scripts/load_test_gunicorn.py
"""

import concurrent.futures
import statistics
import sys
import time
from typing import Dict, List

import requests

BASE_URL = "http://127.0.0.1:8765"
ITERATIONS_PER_WORKER = 20   # requests each worker thread sends
CONCURRENCY_LEVELS = [5, 20]


def ping_server(base_url: str, timeout: int = 5) -> bool:
    try:
        r = requests.get(f"{base_url}/healthz", timeout=timeout)
        return r.status_code < 500
    except requests.RequestException:
        return False


def get_authenticated_session(base_url: str) -> requests.Session | None:
    """Login and return a session cookie jar. Falls back to generating a session cookie if HTTP login is rate-limited."""
    s = requests.Session()
    try:
        # Try direct HTTP form login first with CSRF extraction
        r_get = s.get(f"{base_url}/login", timeout=5)
        import re
        match = re.search(r'name="csrf_token"\s+value="([^"]+)"', r_get.text)
        csrf = match.group(1) if match else ""
        resp = s.post(
            f"{base_url}/login",
            data={"username": "bench_admin", "password": "password123", "csrf_token": csrf},
            timeout=10,
            allow_redirects=True,
        )
        if resp.status_code < 400 and "/login" not in resp.url:
            return s
    except requests.RequestException as e:
        print(f"  HTTP login attempt failed: {e}", file=sys.stderr)

    # Fallback: Generate Flask session cookie directly using app context
    try:
        import os
        from pathlib import Path
        project_dir = str(Path(__file__).resolve().parent.parent)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
        os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "postgresql://mathu@localhost/hospital_db")
        from app import app
        from departments.models.user import User
        with app.app_context():
            with app.test_client() as c:
                u = User.query.filter_by(username="bench_admin").first()
                if not u:
                    from werkzeug.security import generate_password_hash

                    from extensions import db
                    u = User(username="bench_admin", password=generate_password_hash("password123", method="pbkdf2:sha256"), role="admin")
                    db.session.add(u)
                    db.session.commit()
                with c.session_transaction() as sess:
                    sess["_user_id"] = str(u.id)
                    sess["_fresh"] = True
                c.get("/")
                cookie = c.get_cookie("session")
                if cookie:
                    s.cookies.set("session", cookie.value)
                    return s
    except Exception as e:
        print(f"  Session cookie fallback failed: {e}", file=sys.stderr)
    return None


def benchmark_endpoint(
    base_url: str,
    method: str,
    path: str,
    concurrency: int,
    iterations: int,
    session: requests.Session | None = None,
    post_data: dict | None = None,
) -> Dict:
    latencies: List[float] = []
    errors: int = 0

    def worker(_):
        nonlocal errors
        s = session or requests.Session()
        worker_latencies = []
        for _ in range(iterations):
            t0 = time.monotonic()
            try:
                if method == "POST":
                    r = s.post(f"{base_url}{path}", data=post_data or {}, timeout=15)
                else:
                    r = s.get(f"{base_url}{path}", timeout=15)
                elapsed = (time.monotonic() - t0) * 1000
                worker_latencies.append(elapsed)
                if r.status_code >= 500:
                    errors += 1
            except requests.RequestException:
                elapsed = (time.monotonic() - t0) * 1000
                worker_latencies.append(elapsed)
                errors += 1
        return worker_latencies

    wall_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, i) for i in range(concurrency)]
        for f in concurrent.futures.as_completed(futures):
            latencies.extend(f.result())
    wall_elapsed = time.monotonic() - wall_start

    total_requests = len(latencies)
    throughput = total_requests / wall_elapsed if wall_elapsed > 0 else 0

    sorted_lat = sorted(latencies)
    p50 = statistics.median(sorted_lat) if sorted_lat else 0
    p95_idx = int(len(sorted_lat) * 0.95)
    p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0
    mean = statistics.mean(sorted_lat) if sorted_lat else 0

    return {
        "endpoint": f"{method} {path}",
        "concurrency": concurrency,
        "total_requests": total_requests,
        "throughput_rps": round(throughput, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "mean_ms": round(mean, 2),
        "errors": errors,
        "error_pct": round(100 * errors / total_requests, 1) if total_requests else 0,
    }


def main():
    print(f"Checking server at {BASE_URL}...")
    if not ping_server(BASE_URL):
        print(f"ERROR: Server not reachable at {BASE_URL}. Start gunicorn first.")
        print("  SQLALCHEMY_DATABASE_URI='postgresql://mathu@localhost/hospital_db' \\")
        print("  venv/bin/gunicorn app:app --workers 4 --bind 127.0.0.1:8765")
        sys.exit(1)
    print("  Server OK\n")

    session = get_authenticated_session(BASE_URL)
    if session is None:
        print("WARNING: Could not authenticate. Unauthenticated requests may 302.")

    endpoints = [
        ("GET",  "/healthz",                          None),
        ("GET",  "/records/search_patients?term=John", None),
        ("GET",  "/admin/analytics",                  None),
        ("GET",  "/medicine/",                        None),
        ("POST", "/billing/pay_bills",                {"amount": "100"}),
    ]

    results = []
    for concurrency in CONCURRENCY_LEVELS:
        print(f"=== Concurrency C={concurrency} ({ITERATIONS_PER_WORKER} req/worker) ===")
        for method, path, post_data in endpoints:
            r = benchmark_endpoint(
                BASE_URL, method, path, concurrency,
                ITERATIONS_PER_WORKER, session=session, post_data=post_data,
            )
            results.append(r)
            print(
                f"  {method:4s} {path:<45s} "
                f"RPS={r['throughput_rps']:6.2f}  "
                f"p95={r['p95_ms']:7.1f}ms  "
                f"mean={r['mean_ms']:7.1f}ms  "
                f"errors={r['error_pct']}%"
            )
        print()

    # Check for the serialization pattern: throughput falling C5→C20
    print("=== Serialization Check (throughput C5 vs C20) ===")
    for method, path, _ in endpoints:
        c5  = next((r for r in results if r["endpoint"] == f"{method} {path}" and r["concurrency"] == 5),  None)
        c20 = next((r for r in results if r["endpoint"] == f"{method} {path}" and r["concurrency"] == 20), None)
        if c5 and c20:
            ratio = c20["throughput_rps"] / c5["throughput_rps"] if c5["throughput_rps"] > 0 else 0
            flag = " ⚠ DEGRADING" if ratio < 0.5 else (" ~ plateau" if ratio < 0.9 else " ✓ scaling")
            print(f"  {method} {path}: C5={c5['throughput_rps']}rps C20={c20['throughput_rps']}rps ratio={ratio:.2f}{flag}")

    return results


if __name__ == "__main__":
    main()
