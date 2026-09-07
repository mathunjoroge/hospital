"""
departments/emergency/routes.py
─────────────────────────────────
Phase D — Break-glass HTTP routes

Routes:
  POST /emergency/break-glass/invoke        — invoke an emergency override
  GET  /emergency/break-glass/status        — check current user's active grants
  POST /emergency/break-glass/<id>/revoke   — manually revoke a grant (own or admin)
  GET  /admin/break-glass                   — admin audit view of all events
"""

import logging
from datetime import timezone

from flask import abort, jsonify, render_template_string, request
from flask_login import current_user, login_required

from departments.emergency.break_glass import (
    expire_stale_grants,
    invoke_break_glass,
)
from departments.models.break_glass import BreakGlassAccessLog
from departments.rbac import roles_required
from extensions import db

from . import bp

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# POST /emergency/break-glass/invoke
# ─────────────────────────────────────────────────────────────────────────────


@bp.route("/emergency/break-glass/invoke", methods=["POST"])
@login_required
def invoke():
    """
    Invoke a break-glass emergency override for the current user.

    JSON body:
        reason       (str, required)  — mandatory clinical justification
        patient_id   (str, optional)  — target patient ID
        resource_type (str, optional) — target resource type
        resource_id  (str, optional)  — target resource ID
        duration_hours (int, optional) — override window in hours (1–8, default 4)
    """
    data = request.get_json(silent=True) or {}
    reason = (data.get("reason") or "").strip()
    if not reason:
        return jsonify(
            {"error": "A clinical reason is required for break-glass access."}
        ), 400

    patient_id = data.get("patient_id")
    resource_type = data.get("resource_type")
    resource_id = data.get("resource_id")
    duration_hours = int(data.get("duration_hours") or 4)
    duration_hours = max(1, min(duration_hours, 8))  # Clamp to 1–8 hours

    try:
        grant = invoke_break_glass(
            reason=reason,
            patient_id=patient_id,
            resource_type=resource_type,
            resource_id=resource_id,
            duration_hours=duration_hours,
        )
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "message": "Break-glass override granted.",
            "grant": grant.to_dict(),
        }
    ), 201


# ─────────────────────────────────────────────────────────────────────────────
# GET /emergency/break-glass/status
# ─────────────────────────────────────────────────────────────────────────────


@bp.route("/emergency/break-glass/status", methods=["GET"])
@login_required
def status():
    """Return the active break-glass grants for the current user."""
    expire_stale_grants()

    from datetime import datetime

    now = datetime.now(timezone.utc)
    grants = (
        db.session.query(BreakGlassAccessLog)
        .filter(
            BreakGlassAccessLog.user_id == current_user.id,
            BreakGlassAccessLog.is_active.is_(True),
            BreakGlassAccessLog.expires_at > now,
        )
        .order_by(BreakGlassAccessLog.invoked_at.desc())
        .all()
    )
    return jsonify({"active_grants": [g.to_dict() for g in grants]}), 200


# ─────────────────────────────────────────────────────────────────────────────
# POST /emergency/break-glass/<int:grant_id>/revoke
# ─────────────────────────────────────────────────────────────────────────────


@bp.route("/emergency/break-glass/<int:grant_id>/revoke", methods=["POST"])
@login_required
def revoke(grant_id):
    """
    Revoke a break-glass grant before its natural expiry.
    Clinicians can revoke their own grants; admins can revoke any grant.
    """
    grant = db.session.get(BreakGlassAccessLog, grant_id)
    if not grant:
        abort(404)

    # Authorisation: own grant or admin
    if grant.user_id != current_user.id and current_user.role != "admin":
        abort(403)

    grant.revoke()
    db.session.commit()

    return jsonify({"message": f"Grant {grant_id} revoked.", "grant_id": grant_id}), 200


# ─────────────────────────────────────────────────────────────────────────────
# GET /admin/break-glass  — admin audit view
# ─────────────────────────────────────────────────────────────────────────────

ADMIN_BREAK_GLASS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Break-glass Access Audit — Admin</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }
    h1 { color: #f87171; margin-bottom: 0.25rem; }
    .subtitle { color: #94a3b8; margin-bottom: 1.5rem; font-size: 0.9rem; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th { background: #1e293b; padding: 0.6rem 0.75rem; text-align: left; color: #7dd3fc; border-bottom: 2px solid #334155; }
    td { padding: 0.55rem 0.75rem; border-bottom: 1px solid #1e293b; vertical-align: top; }
    tr:hover td { background: #1e293b; }
    .badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    .active   { background: #dc2626; color: #fff; }
    .expired  { background: #374151; color: #9ca3af; }
    .notified { background: #065f46; color: #a7f3d0; }
    .reason   { max-width: 280px; word-break: break-word; }
  </style>
</head>
<body>
  <h1>⚠️ Break-glass Access Audit</h1>
  <p class="subtitle">All emergency override invocations — most recent first. This log is append-only.</p>
  {% if logs %}
  <table>
    <thead>
      <tr>
        <th>#</th><th>Invoked At</th><th>User</th><th>Role</th><th>Patient ID</th>
        <th>Reason</th><th>Expires At</th><th>Status</th><th>Supervisor Notified</th>
      </tr>
    </thead>
    <tbody>
    {% for log in logs %}
      <tr>
        <td>{{ log.id }}</td>
        <td>{{ log.invoked_at.strftime('%Y-%m-%d %H:%M UTC') if log.invoked_at else '—' }}</td>
        <td>{{ log.username }}</td>
        <td>{{ log.user_role }}</td>
        <td>{{ log.patient_id or '—' }}</td>
        <td class="reason">{{ log.reason }}</td>
        <td>{{ log.expires_at.strftime('%Y-%m-%d %H:%M UTC') if log.expires_at else '—' }}</td>
        <td>
          <span class="badge {{ 'active' if log.is_valid() else 'expired' }}">
            {{ 'ACTIVE' if log.is_valid() else 'EXPIRED' }}
          </span>
        </td>
        <td>
          <span class="badge {{ 'notified' if log.supervisor_notified else 'expired' }}">
            {{ '✓ YES' if log.supervisor_notified else 'NO' }}
          </span>
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p style="color:#64748b">No break-glass events recorded.</p>
  {% endif %}
</body>
</html>
"""


@bp.route("/admin/break-glass", methods=["GET"])
@login_required
@roles_required("admin")
def admin_audit():
    """Admin-only view of all break-glass override events."""
    expire_stale_grants()

    if request.args.get("format") == "json":
        logs = BreakGlassAccessLog.query.order_by(
            BreakGlassAccessLog.invoked_at.desc()
        ).all()
        return jsonify({"break_glass_events": [log.to_dict() for log in logs]}), 200

    logs = (
        BreakGlassAccessLog.query.order_by(BreakGlassAccessLog.invoked_at.desc())
        .limit(200)
        .all()
    )
    return render_template_string(ADMIN_BREAK_GLASS_TEMPLATE, logs=logs), 200
