from flask import jsonify
from flask_login import login_required

from . import bp


@bp.route("/")
@login_required
def index():
    return "Analytics & KPI Module Active - Phase 6 MVP"


@bp.route("/api/dashboard", methods=["GET"])
@login_required
def get_executive_dashboard():
    """
    Stub for fetching the latest KPI snapshots for the admin dashboard.
    """
    return jsonify(
        {
            "status": "success",
            "message": "Executive dashboard data endpoint active.",
            "data": {
                "today_outpatient_visits": 0,
                "today_admissions": 0,
                "bed_occupancy_rate": "0%",
            },
        }
    )
