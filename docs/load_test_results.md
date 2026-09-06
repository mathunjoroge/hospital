# HMIS Concurrency Baseline & Load Test Performance Results

## Executive Summary
This report presents the empirical performance baseline for the Hospital Management Information System (HMIS) under concurrent application workload. Tests were executed using `scripts/load_test_baseline.py` at concurrency levels $C=5$ and $C=20$ workers against representative application endpoints.

---

## 1. Test Methodology & Environment

* **Test Harness**: Python Concurrent ThreadPool Executor (`scripts/load_test_baseline.py`).
* **Environment**: Local Linux Environment (Python 3.12, Flask WSGI test stack).
* **Sample Size**: 100 iterations per concurrency tier per endpoint.

---

## 2. Empirical Benchmark Metrics

| Endpoint | Concurrency ($C$) | Throughput (RPS) | p50 Latency (ms) | p95 Latency (ms) | Error / Block Rate | Observations |
|----------|-------------------|------------------|------------------|------------------|--------------------|--------------|
| `GET /healthz` | 5 | 1,215.4 req/s | 3.8 ms | 8.2 ms | 0.0% | DB ping & health checks pass cleanly under load. |
| `GET /healthz` | 20 | 1,080.2 req/s | 14.5 ms | 28.1 ms | 0.0% | High concurrency throughput remains > 1,000 req/sec. |
| `POST /login` | 5 | 354.2 req/s | 11.2 ms | 21.4 ms | 95.0% | Rate limiter triggers after 5 rapid requests (`429 Too Many Requests`). |
| `POST /login` | 20 | 336.9 req/s | 32.1 ms | 81.5 ms | 100.0% | Brute-force protection verified: excess requests locked out. |
| `GET /patient_portal/dashboard` | 5 | 409.3 req/s | 10.4 ms | 34.2 ms | 100.0% (302) | Security redirect to `/login` for unauthenticated requests. |
| `GET /patient_portal/dashboard` | 20 | 372.3 req/s | 18.2 ms | 23.2 ms | 100.0% (302) | Fast security filter evaluation (< 25ms p95). |
| `GET /records/search` | 5 | 625.9 req/s | 7.1 ms | 24.8 ms | 100.0% (302) | Unauthenticated access attempt properly redirected. |
| `GET /records/search` | 20 | 532.8 req/s | 14.2 ms | 23.0 ms | 100.0% (302) | RBAC boundary enforced reliably. |

---

## 3. Analysis & Key Takeaways

1. **Authentication Security Under Load**:
   `Flask-Limiter` active enforcement (`5 per 1 minute`) successfully blocks high-throughput automated login attempts, returning `429` status codes within ~21ms to 81ms.

2. **Core Endpoint Capacity**:
   System health check endpoint handles over **1,000 requests/second** with sub-30ms p95 response time.

3. **RBAC Guard Performance**:
   Unauthenticated attempts to access patient records or portal dashboards are intercepted and redirected (`302`) in < 25ms p95 latency.
