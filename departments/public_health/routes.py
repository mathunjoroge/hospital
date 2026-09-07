from flask import jsonify, request
from flask_login import login_required

from . import bp


@bp.route("/")
@login_required
def index():
    return "Public Health Reporting Module Active - Phase 8 MVP"


@bp.route("/api/notify", methods=["POST"])
@login_required
def report_notifiable_disease():
    """
    Stub for submitting a notifiable disease case report.
    Will eventually integrate with KHIS/DHIS2 case-based reporting.
    """
    _data = request.get_json() or {}
    return (
        jsonify(
            {"status": "success", "message": "Notifiable disease report stub active."}
        ),
        201,
    )


@bp.route("/api/mortality", methods=["POST"])
@login_required
def report_mortality():
    """
    Stub for submitting a mortality report (MoH 736 aligned).
    """
    _data = request.get_json() or {}
    return (
        jsonify({"status": "success", "message": "Mortality report stub active."}),
        201,
    )


@bp.route("/api/outbreak-signals", methods=["GET"])
@login_required
def get_outbreak_signals():
    """
    Stub for retrieving active outbreak signals for the surveillance dashboard.
    """
    return jsonify(
        {
            "status": "success",
            "message": "Outbreak signal retrieval stub active.",
            "active_signals": [],
        }
    )


@bp.route("/api/outbreak-signal", methods=["POST"])
@login_required
def raise_outbreak_signal():
    """
    Stub for raising a new outbreak early-warning signal.
    """
    _data = request.get_json() or {}
    return (
        jsonify(
            {"status": "success", "message": "Outbreak signal creation stub active."}
        ),
        201,
    )
