"""
departments/compliance/routes.py
──────────────────────────────────
API and UI routes for HIPAA & HITRUST CSF Compliance & Cryptographic Audit Verification.
"""

from flask import jsonify, render_template
from flask_login import login_required

from departments.audit import verify_audit_log_chain
from departments.compliance.hipaa_engine import HIPAAComplianceEngine
from departments.rbac import roles_required

from . import compliance_bp


@compliance_bp.route("/hipaa-dashboard", methods=["GET"])
@login_required
@roles_required("admin", "api")
def hipaa_dashboard_view():
    """Render HITRUST CSF & HIPAA Readiness Dashboard UI."""
    audit_report = HIPAAComplianceEngine.run_full_hipaa_audit()
    chain_status = verify_audit_log_chain()

    return render_template(
        "compliance/hipaa_dashboard.html",
        report=audit_report,
        chain_status=chain_status,
    )


@compliance_bp.route("/api/hipaa-status", methods=["GET"])
@login_required
@roles_required("admin", "api")
def get_hipaa_status():
    """Fetch automated HIPAA / HITRUST CSF 5-domain audit status."""
    report = HIPAAComplianceEngine.run_full_hipaa_audit()
    return jsonify({"status": "success", "report": report}), 200


@compliance_bp.route("/api/verify-audit-chain", methods=["GET"])
@login_required
@roles_required("admin", "api")
def api_verify_audit_chain():
    """Verify cryptographic SHA-256 hash chain of all audit log entries."""
    res = verify_audit_log_chain()
    return jsonify({"status": "success", "verification": res}), 200
