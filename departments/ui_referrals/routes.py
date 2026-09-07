"""
Referrals UI routes.
"""

from flask import render_template
from flask_login import login_required

from . import bp


@bp.route("/handover")
@login_required
def referral_handover():
    return render_template("referrals/handover.html")
