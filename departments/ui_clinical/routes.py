"""
Clinical UI routes.
"""

from flask import render_template
from flask_login import login_required

from . import bp


@bp.route("/prescribe")
@login_required
def prescription_workbench():
    """
    Renders the interactive prescription safety and consent workbench.
    """
    return render_template("clinical/prescribe.html")
