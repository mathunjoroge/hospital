"""
MCH UI routes.
"""

from flask import render_template
from flask_login import login_required

from departments.rbac import roles_required

from . import bp


@bp.route("/workbench")
@login_required
@roles_required("mch", "nursing", "doctor", "medicine", "admin")
def mch_workbench():
    from departments.mch.cold_chain import ColdChainEngine

    cc = ColdChainEngine()
    stock_summary = cc.get_stock_summary()
    near_expiry = cc.get_near_expiry_alerts(30)
    return render_template(
        "mch/workbench.html",
        stock_summary=stock_summary,
        near_expiry=near_expiry,
    )
