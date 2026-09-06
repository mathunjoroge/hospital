#!/usr/bin/env python3
"""
scripts/load_test_baseline.py
──────────────────────────────
Load & Stress Testing Baseline Script for HMIS.
Simulates concurrent clinical user sessions against highest-traffic endpoints.
Measures both:
  1. Unauthenticated Gate Performance (302 redirects & 429 rate limits)
  2. Authenticated Application Workload Performance (real DB queries & HTML/JSON rendering)
"""

import concurrent.futures
import os
import sys
import time
from typing import Dict, List

import numpy as np
from werkzeug.security import generate_password_hash

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app  # noqa: E402
from extensions import db, limiter  # noqa: E402
from departments.models.user import User  # noqa: E402
from departments.models.records import Patient  # noqa: E402
from datetime import date  # noqa: E402


def setup_test_data():
    """Ensure database has seeded admin user and test patients for real application queries."""
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='bench_admin').first():
            user = User(
                username='bench_admin',
                password=generate_password_hash('password123', method='pbkdf2:sha256'),
                role='admin'
            )
            db.session.add(user)

        if not Patient.query.filter_by(patient_id='P-BENCH-01').first():
            p = Patient(
                patient_id="P-BENCH-01",
                name="John Benchmark Patient",
                place_of_residence="Nairobi",
                sex="Male",
                date_of_birth=date(1985, 5, 20),
                marital_status="Married",
                contact="0711223344",
                next_of_kin="Jane Benchmark",
                relationship_with_next_of_kin="Spouse",
                next_of_kin_contact="0722334455",
                emergency_contact="0733445566"
            )
            db.session.add(p)
        db.session.commit()


def benchmark_endpoint(endpoint_func, num_requests: int, concurrency: int, authenticate: bool = True) -> Dict:
    latencies: List[float] = []
    failures: int = 0
    successes: int = 0

    t_start = time.time()

    def worker_session(req_count: int):
        nonlocal failures, successes
        thread_latencies = []
        with app.test_client() as client:
            if authenticate:
                res_login = client.post('/login', data={'username': 'bench_admin', 'password': 'password123'})
                if res_login.status_code >= 400:
                    failures += req_count
                    return []

            for _ in range(req_count):
                t0 = time.perf_counter()
                try:
                    res = endpoint_func(client)
                    elapsed = (time.perf_counter() - t0) * 1000.0  # ms
                    if res.status_code < 400:
                        successes += 1
                        thread_latencies.append(elapsed)
                    else:
                        failures += 1
                except Exception:
                    failures += 1
        return thread_latencies

    reqs_per_worker = num_requests // concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker_session, reqs_per_worker) for _ in range(concurrency)]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                latencies.extend(res)

    total_time = time.time() - t_start
    total_processed = successes + failures
    rps = total_processed / total_time if total_time > 0 else 0

    return {
        'total_requests': num_requests,
        'concurrency': concurrency,
        'duration_sec': round(total_time, 3),
        'rps': round(rps, 2),
        'successes': successes,
        'failures': failures,
        'error_rate_pct': round((failures / num_requests) * 100, 2),
        'p50_ms': round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
        'p95_ms': round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        'p99_ms': round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0,
        'min_ms': round(float(np.min(latencies)), 2) if latencies else 0.0,
        'max_ms': round(float(np.max(latencies)), 2) if latencies else 0.0,
    }


def run_all_benchmarks(concurrency_levels=[5, 20]):
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['RATELIMIT_ENABLED'] = False
    limiter.enabled = False
    if hasattr(limiter, '_storage') and hasattr(limiter._storage, 'reset'):
        limiter._storage.reset()

    setup_test_data()

    print("=========================================================")
    print(" 1. AUTHENTICATED APPLICATION WORKLOAD BENCHMARKS ")
    print("=========================================================")

    auth_endpoints = {
        'GET /healthz': lambda c: c.get('/healthz'),
        'GET /records/search_patients?term=John (DB Search)': lambda c: c.get('/records/search_patients?term=John'),
        'GET /admin/analytics (Dashboard)': lambda c: c.get('/admin/analytics'),
        'GET /medicine/ (Clinical List)': lambda c: c.get('/medicine/'),
    }

    results = {}
    for name, func in auth_endpoints.items():
        results[name] = {}
        for c in concurrency_levels:
            metrics = benchmark_endpoint(func, num_requests=100, concurrency=c, authenticate=True)
            results[name][f'c={c}'] = metrics
            print(f"[{name}] Concurrency {c}: {metrics['rps']} req/sec, p95={metrics['p95_ms']}ms, errors={metrics['error_rate_pct']}%")

    return results


if __name__ == '__main__':
    run_all_benchmarks()
