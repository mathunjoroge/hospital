#!/usr/bin/env python3
"""
scripts/load_test_baseline.py
──────────────────────────────
Load & Stress Testing Baseline Script for HMIS.
Simulates concurrent clinical user sessions against highest-traffic endpoints:
  - GET /healthz
  - POST /login
  - GET /patient_portal/dashboard
  - GET /records/patients
"""

import concurrent.futures
import os
import sys
import time
from typing import Dict, List

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app  # noqa: E402


def benchmark_endpoint(endpoint_func, num_requests: int, concurrency: int) -> Dict:
    latencies: List[float] = []
    failures: int = 0
    successes: int = 0

    t_start = time.time()

    def worker():
        nonlocal failures, successes
        with app.test_client() as client:
            t0 = time.perf_counter()
            try:
                res = endpoint_func(client)
                elapsed = (time.perf_counter() - t0) * 1000.0  # ms
                if res.status_code < 400:
                    successes += 1
                else:
                    failures += 1
                return elapsed
            except Exception:
                failures += 1
                return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res is not None:
                latencies.append(res)

    total_time = time.time() - t_start
    rps = num_requests / total_time if total_time > 0 else 0

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

    endpoints = {
        'GET /healthz': lambda c: c.get('/healthz'),
        'POST /login (Auth)': lambda c: c.post('/login', data={'username': 'admin', 'password': 'password123'}),
        'GET /patient_portal/dashboard': lambda c: c.get('/patient_portal/dashboard'),
        'GET /records/search': lambda c: c.get('/records/patients?q=John'),
    }

    results = {}
    for name, func in endpoints.items():
        results[name] = {}
        for c in concurrency_levels:
            metrics = benchmark_endpoint(func, num_requests=100, concurrency=c)
            results[name][f'c={c}'] = metrics
            print(f"[{name}] Concurrency {c}: {metrics['rps']} req/sec, p95={metrics['p95_ms']}ms, errors={metrics['error_rate_pct']}%")

    return results


if __name__ == '__main__':
    run_all_benchmarks()
