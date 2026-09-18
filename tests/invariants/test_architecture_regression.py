"""
tests/invariants/test_architecture_regression.py
──────────────────────────────────────────────────
Architecture regression firewall.

These tests scan the source code AST and text for patterns that were either
found in P0 security remediation or are known to reintroduce vulnerabilities.
They run in CI and catch regressions introduced by future commits or AI agents.

DO NOT remove or weaken these tests without a security review sign-off.
"""

import ast
import os
import re
from pathlib import Path

DEPT_ROOT = Path(__file__).parent.parent.parent / "departments"


def _python_files():
    for root, dirs, files in os.walk(DEPT_ROOT):
        dirs[:] = [d for d in dirs if "__pycache__" not in d]
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ─── identity fallback pattern ─────────────────────────────────────────────────


class TestNoIdentityFallback:
    """
    INVARIANT: session.get("user_id", 1) — or any integer fallback — must
    never appear in route handlers.  This was the root cause of the P0
    identity-spoofing vulnerability.
    """

    PATTERN = re.compile(
        r'session\.get\(\s*["\']user_id["\']\s*,\s*\d+\s*\)',
    )

    def test_no_integer_fallback_on_session_user_id(self):
        violations = []
        for path in _python_files():
            src = _source(path)
            for lineno, line in enumerate(src.splitlines(), 1):
                stripped = line.strip()
                # Skip pure comment lines and docstring references
                if (
                    stripped.startswith("#")
                    or stripped.startswith('"""')
                    or stripped.startswith("- ")
                ):
                    continue
                if self.PATTERN.search(line):
                    violations.append(
                        f"{path.relative_to(DEPT_ROOT.parent)}:{lineno}: {stripped}"
                    )

        assert not violations, (
            "ARCHITECTURE REGRESSION: session.get('user_id', <default>) found. "
            "User identity must always come from current_user, never have an integer fallback.\n"
            + "\n".join(violations)
        )


# ─── float in financial models ─────────────────────────────────────────────────


class TestNoFloatFinancialColumns:
    """
    INVARIANT: financial columns must use db.Numeric (or Decimal), never db.Float.
    Float arithmetic produces rounding errors that compound in billing and payroll.
    """

    # Keywords that strongly indicate a financial column (not clinical measurements)
    FINANCIAL_KEYWORDS = re.compile(
        r"\b(unit_cost|buying_price|selling_price|unit_price|total_cost|gross_pay|"
        r"net_pay|total_deductions|clinic_fee|allowance_value|deduction_value|"
        r"invoice_total|amount_paid|balance|grand_total|subtotal)\b",
        re.IGNORECASE,
    )
    # Also catch generic financial names on the column name (before the = sign)
    FINANCIAL_COLNAME = re.compile(
        r"^\s+(\w+)\s*=\s*db\.Column\s*\(\s*db\.Float",
    )
    FINANCIAL_COLNAME_KEYWORDS = re.compile(
        r"\b(price|cost|fee|pay|salary|allowance|deduction|charge|invoice)\b",
        re.IGNORECASE,
    )
    FLOAT_COLUMN = re.compile(r"db\.Column\s*\(\s*db\.Float")

    def test_no_float_on_financial_columns(self):
        violations = []
        for path in _python_files():
            if "models" not in str(path):
                continue
            src = _source(path)
            for lineno, line in enumerate(src.splitlines(), 1):
                if not self.FLOAT_COLUMN.search(line):
                    continue
                # Match on explicit financial keyword names in the full line comment/name
                if self.FINANCIAL_KEYWORDS.search(line):
                    violations.append(
                        f"{path.relative_to(DEPT_ROOT.parent)}:{lineno}: {line.strip()}"
                    )
                    continue
                # Also catch column variable names that are financial
                m = self.FINANCIAL_COLNAME.match(line)
                if m and self.FINANCIAL_COLNAME_KEYWORDS.search(m.group(1)):
                    violations.append(
                        f"{path.relative_to(DEPT_ROOT.parent)}:{lineno}: {line.strip()}"
                    )

        assert not violations, (
            "ARCHITECTURE REGRESSION: db.Float found on a financial column.\n"
            "Use db.Numeric(precision, scale) instead.\n" + "\n".join(violations)
        )


# ─── unauthenticated route detection ──────────────────────────────────────────

# Routes that are explicitly permitted to be unauthenticated (login page, static assets, etc.)
ALLOWED_UNAUTH_ROUTES = {
    # Auth flows
    "login",
    "logout",
    "register",
    "reset_password",
    "confirm_email",
    "patient_login",
    "patient_register",
    "patient_forgot_password",
    "patient_reset_password",
    "verify_mfa",
    "oauth_callback",
    # Health checks / static
    "health",
    "healthz",
    "favicon",
    "static",
    # FHIR capability statement (public by spec)
    "fhir_capability",
    "capability_statement",
    # SSO entry points
    "sso_login",
    "sso_callback",
    "oidc_callback",
    "ldap_login",
}

CLINICAL_WRITE_KEYWORDS = re.compile(
    r"\b(chart|dispense|administer|prescribe|admit|discharge|bill|auto_bill|"
    r"save_chemo|create_order|record_result|verify_result)\b",
    re.IGNORECASE,
)


class TestClinicalWriteRoutesRequireAuth:
    """
    INVARIANT: any route function whose name suggests a clinical write action
    must have a @login_required decorator (or equivalent).

    This is a heuristic scan — it catches obvious cases. Full coverage is
    verified by the explicit tests in test_clinical_auth_invariants.py.
    """

    def _route_functions_without_auth(self):
        """Yield (path, lineno, func_name) for clinical routes missing auth."""
        auth_decorators = {
            "login_required",
            "roles_required",
            "require_role",
            "jwt_required",
            "token_required",
            "patient_login_required",
        }
        for path in _python_files():
            try:
                tree = ast.parse(_source(path))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue

                dec_names = set()
                for d in node.decorator_list:
                    if isinstance(d, ast.Call):
                        fn = d.func
                        if isinstance(fn, ast.Attribute):
                            dec_names.add(fn.attr)
                        elif isinstance(fn, ast.Name):
                            dec_names.add(fn.id)
                    elif isinstance(d, ast.Attribute):
                        dec_names.add(d.attr)
                    elif isinstance(d, ast.Name):
                        dec_names.add(d.id)

                has_route = any("route" in n for n in dec_names)
                has_auth = bool(dec_names & auth_decorators)
                is_clinical_write = bool(CLINICAL_WRITE_KEYWORDS.search(node.name))
                is_allowed = node.name in ALLOWED_UNAUTH_ROUTES

                if has_route and not has_auth and is_clinical_write and not is_allowed:
                    yield path, node.lineno, node.name

    def test_clinical_write_routes_have_auth(self):
        violations = [
            f"{p.relative_to(DEPT_ROOT.parent)}:{ln} — {fn}()"
            for p, ln, fn in self._route_functions_without_auth()
        ]
        assert not violations, (
            "ARCHITECTURE REGRESSION: clinical-write route(s) found without authentication:\n"
            + "\n".join(violations)
            + "\nAdd @login_required and @roles_required to each."
        )


# ─── stock constraint validation ───────────────────────────────────────────────


class TestStockCheckConstraintsDefined:
    """
    INVARIANT: Drug and Batch models must define CheckConstraints preventing
    negative stock quantities at the database level.
    """

    def test_drug_model_has_non_negative_stock_constraint(self):
        from departments.models.pharmacy import Drug

        constraint_names = {
            c.name for c in Drug.__table__.constraints if hasattr(c, "name") and c.name
        }
        assert "ck_drug_stock_non_negative" in constraint_names, (
            "ARCHITECTURE REGRESSION: Drug model is missing the "
            "'ck_drug_stock_non_negative' CheckConstraint. "
            "Without it, a bug in dispensing logic can store negative stock."
        )

    def test_batch_model_has_non_negative_stock_constraint(self):
        from departments.models.pharmacy import Batch

        constraint_names = {
            c.name for c in Batch.__table__.constraints if hasattr(c, "name") and c.name
        }
        assert "ck_batch_stock_non_negative" in constraint_names, (
            "ARCHITECTURE REGRESSION: Batch model is missing the "
            "'ck_batch_stock_non_negative' CheckConstraint."
        )


# ─── debug print detection ────────────────────────────────────────────────────


class TestNoDebugPrintsInRoutes:
    """
    INVARIANT: route handler files must not contain print() calls.
    Debug prints bypass structured logging, can leak sensitive state to stdout,
    and indicate incomplete cleanup of development code.
    """

    PRINT_PATTERN = re.compile(r"^\s*print\s*\(", re.MULTILINE)

    def test_no_print_in_route_files(self):
        violations = []
        for path in _python_files():
            if "routes" not in path.name and "views" not in path.name:
                continue
            src = _source(path)
            for lineno, line in enumerate(src.splitlines(), 1):
                if re.match(r"\s*print\s*\(", line) and not line.strip().startswith(
                    "#"
                ):
                    violations.append(
                        f"{path.relative_to(DEPT_ROOT.parent)}:{lineno}: {line.strip()}"
                    )

        assert not violations, (
            "DEBUG CODE IN PRODUCTION: print() calls found in route files. "
            "Use logger.debug/info/error instead.\n" + "\n".join(violations)
        )
