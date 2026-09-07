"""
MCH UI routes.
"""

from flask import render_template
from flask_login import login_required

from . import bp


@bp.route("/workbench")
@login_required
def mch_workbench():
    return render_template("mch/workbench.html")
