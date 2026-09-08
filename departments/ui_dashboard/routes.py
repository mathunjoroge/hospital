"""
UI Dashboard routes. Renders the frontend templates.
"""

from flask import render_template
from flask_login import login_required

from departments.rbac import roles_required

from . import bp


@bp.route("/dashboard/queue/<int:provider_id>")
@login_required
@roles_required("doctor", "nursing")
def live_queue_dashboard(provider_id: int):
    """
    Renders the real-time waiting room dashboard for a specific provider.
    """
    return render_template("dashboard/queue.html", provider_id=provider_id)
