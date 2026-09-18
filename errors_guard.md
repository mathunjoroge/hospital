# errors_guard.md

Anti-pattern reference for agents working on `mathunjoroge/hospital`.
Every section documents a bug that was introduced and later found in CI.
Read this before touching imports, datetimes, logging, or running ruff.

---

## 1. The inline-noqa comment that swallows a module import

### What happened

Three `departments/*/\_\_init\_\_.py` files had the `# noqa: F401` comment
placed on the same line as the first module in a multi-line import, making
that first module part of the comment and never imported:

```python
# BROKEN — chat_bot is part of the comment, not the import
from . import (  # noqa: F401    chat_bot,
    consultations,
    inpatients,
)
```

`chat_bot` was never imported, so `medicine.chatbot_interface` was never
registered as a Flask route. Every template that called
`url_for('medicine.chatbot_interface')` raised a `BuildError` at runtime,
which crashed whichever test first loaded a page containing that URL — in
this case `test_admin_analytics_route_rbac`, several hundred lines away from
the broken import.

### Files that had this bug

| File | Swallowed module |
|---|---|
| `departments/medicine/__init__.py` | `chat_bot` |
| `departments/pharmacy/__init__.py` | `ai_discovery` |
| `departments/api/__init__.py` | `auth` |

### The fix

Move the `# noqa: F401` comment to the end of the opening paren line with
no module name after it. Each import goes on its own line:

```python
# CORRECT
from . import (  # noqa: F401
    chat_bot,
    consultations,
    inpatients,
)
```

### Detection command

Run this before every commit to catch the pattern:

```bash
grep -rn "# noqa: F401 " departments/*/\_\_init\_\_.py
```

Any hit where the `# noqa` is followed by a module name (rather than a
newline) is broken. The rule of thumb: `# noqa` must be the **last thing on
the line**; nothing executable may follow it.

---

## 2. SQLite strips timezone info — never compare naive and aware datetimes

### What happened

`PatientUser.reset_token_expiry` is stored with `db.Column(db.DateTime(timezone=True))`.
In production (Postgres) that works correctly. Under test with SQLite, timezone
info is silently stripped on the database roundtrip. The test then compared the
naive value coming back from SQLite against `datetime.now(timezone.utc)` (which
is tz-aware), raising:

```
TypeError: can't compare offset-naive and offset-aware datetimes
```

### Rule

**Never compare a column value loaded from SQLite directly against a
tz-aware datetime.** Always normalise the DB value first.

The helper already exists in `departments/patient_portal/auth.py`:

```python
def _as_utc(dt: "datetime | None") -> "datetime | None":
    """
    SQLite strips timezone info on roundtrip even with DateTime(timezone=True).
    Any naive datetime stored by this app was written as UTC, so we re-attach
    UTC here before comparing against datetime.now(timezone.utc).
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
```

Use it in tests and in any production code that reads `DateTime` columns
back from the DB:

```python
# CORRECT
expiry = _as_utc(pu.reset_token_expiry)
assert expiry > datetime.now(timezone.utc)

# BROKEN
assert pu.reset_token_expiry > datetime.now(timezone.utc)
```

### Rule for writing datetimes

Always write with an explicit UTC timezone. Never use the naive forms:

```python
# CORRECT
datetime.now(timezone.utc)

# BROKEN — all of these produce naive datetimes
datetime.utcnow()
datetime.today()
datetime.now()
date.today()
```

---

## 3. ruff --fix can eat `pass` statements and break syntax

### What happened

Running `ruff --fix` (especially with `--unsafe-fixes`) on except blocks
sometimes merged a `pass` statement onto the same line as a `# noqa` comment,
producing:

```python
except Exception:  # noqa: BLE001  pass
```

This is a syntax error. Python sees the `pass` as part of the comment.

### Rule

After any `ruff --fix` run, always verify with:

```bash
python -m py_compile app.py
python -m compileall departments/ -q
```

Then re-run `ruff check` to confirm the fix didn't introduce new violations.
A CI-safe sequence:

```bash
ruff check . --select E,F,W,I --ignore E501          # baseline
ruff check . --select E,F,W,I --ignore E501 --fix    # auto-fix
python -m compileall . -q                             # syntax check
ruff check . --select E,F,W,I --ignore E501          # confirm clean
```

---

## 4. E402 — module-level imports not at top of file

### Context

Every `departments/*/\_\_init\_\_.py` registers a Flask Blueprint, then imports
submodules (routes, models) **after** the Blueprint exists. This is intentional
— the submodules depend on `bp` being defined first. These late imports are
**not** a bug and must not be moved to the top.

E402 is suppressed for all affected files via `pyproject.toml`:

```toml
[tool.ruff.lint.per-file-ignores]
"departments/medicine/__init__.py" = ["E402"]
"departments/pharmacy/__init__.py" = ["E402"]
# ... and every other departments/<name>/__init__.py
```

### Rules

- Do not add `# noqa: E402` inline comments to suppress this — they get
  stripped by `ruff --fix` on the next pass, causing CI to fail again.
- Do not move the submodule imports above the `bp = Blueprint(...)` line.
- When adding a **new** department, add its `__init__.py` to the
  `per-file-ignores` list in `pyproject.toml` immediately.

---

## 5. Logging — do not use `.error(..., exc_info=True)` inside except blocks

### Rule

`logger.error("msg", exc_info=True)` inside an `except` block is equivalent
to `logger.exception("msg")` but triggers ruff G201. Use `.exception()` instead:

```python
# CORRECT
try:
    ...
except Exception:  # noqa: BLE001
    logger.exception("Error in admin.index")

# BROKEN — ruff G201
try:
    ...
except Exception as e:
    logger.error("Error in admin.index: %s", e, exc_info=True)
```

Also avoid passing the exception variable as a format argument inside
`.exception()` — ruff TRY401 flags it because the traceback already contains
the exception text:

```python
# CORRECT
logger.exception("Error in billing.submit")

# BROKEN — ruff TRY401 (redundant)
logger.exception("Error in billing.submit: %s", e)
```

---

## 6. Broad except — BLE001

### Context

`except Exception` is flagged as BLE001. In most cases in this codebase the
broad catch is intentional (Flask route handlers that must return a response
rather than propagate). Suppress per-occurrence with a comment:

```python
except Exception:  # noqa: BLE001
    logger.exception("Route handler fallback")
    return jsonify({"error": "Internal error"}), 500
```

### Rule

Never suppress BLE001 on an entire file. Suppress only on the specific line
where the broad catch is intentional, and add a comment explaining why.

---

## 7. Dependency audit — aiohttp and pip-audit

### What happened

`aiohttp==3.9.5` accumulated 34 CVEs. pip-audit caught them in CI but the
workflow's `--ignore-vuln` list did not include them, so the gate failed.

### Current pinned version

`aiohttp==3.14.3` — do not downgrade.

### Rule

When pip-audit reports new vulnerabilities, always prefer upgrading the
package over adding `--ignore-vuln` entries. Add an `--ignore-vuln` only
when:

- The package cannot be upgraded without breaking something, **and**
- The vulnerability does not apply to this app's usage (document why in
  the workflow comment).

The current ignored advisories are listed with justifications in
`.github/workflows/ci.yml`. Do not add new entries silently.

---

## 8. CI lint command — exact invocation

The CI runs:

```bash
ruff check . --select E,F,W,I --ignore E501
```

This selects only `E` (pycodestyle errors), `F` (Pyflakes), `W` (pycodestyle
warnings), and `I` (isort). Rules outside these categories (e.g. `BLE`, `G`,
`DTZ`, `RUF`, `TRY`, `SIM`) are **not checked by CI** but may still be
present in the codebase as suppressed warnings from earlier full-ruleset runs.

Always test your changes with the exact CI command before pushing:

```bash
ruff check . --select E,F,W,I --ignore E501
```

Running `ruff check .` without `--select` will flag many more rules and give
a misleading picture of what will block CI.

---

## 9. Quick pre-push checklist

```bash
# 1. Confirm no module-name text appears after any noqa comment
grep -rn "# noqa: F401 " departments/*/\_\_init\_\_.py

# 2. Run the exact CI lint command
ruff check . --select E,F,W,I --ignore E501

# 3. Syntax-check everything
python -m compileall . -q 2>&1 | grep -v "^Listing"

# 4. Run tests with the CI coverage threshold
FLASK_ENV=testing \
SECRET_KEY=ci-test-secret \
REDIS_HOST=localhost \
REDIS_PORT=6379 \
SQLALCHEMY_DATABASE_URI=sqlite:///test.db \
python -m pytest --cov=. --cov-report=term-missing --cov-fail-under=20 -q

# 5. All four must be clean before pushing.
```
