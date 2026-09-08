"""
UI Dashboard routes. Renders the frontend templates.
"""

from flask import render_template
from flask_login import login_required

from departments.rbac import roles_required

from . import bp


@bp.route("/dashboard/queue/<provider_id>")
@bp.route("/dashboard/queue", defaults={"provider_id": "all"})
@login_required
@roles_required("doctor", "nursing")
def live_queue_dashboard(provider_id: str):
    """
    Renders the real-time waiting room dashboard for a specific provider or all providers.
    """
    return render_template("dashboard/queue.html", provider_id=provider_id)
