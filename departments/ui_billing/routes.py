"""
Billing UI routes.
"""

from flask import render_template
from flask_login import login_required

from departments.rbac import roles_required

from . import bp


@bp.route("/claims")
@login_required
@roles_required("billing", "admin")
def claims_scrubber():
    return render_template("billing/claims.html")
