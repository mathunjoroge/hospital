from flask import jsonify, request
from flask_login import login_required

from . import bp


@bp.route("/")
@login_required
def index():
    return "Offline Sync Engine Active - Phase 7 MVP"


@bp.route("/push", methods=["POST"])
@login_required
def push_offline_changes():
    """
    Receives a batch of offline mutations from the PWA.
    Will eventually process them in a transaction and check for conflicts.
    """
    _data = request.get_json() or {}
    return (
        jsonify(
            {
                "status": "success",
                "message": "Sync push endpoint active. Ready to process queue.",
                "processed_count": 0,
                "conflict_count": 0,
            }
        ),
        200,
    )


@bp.route("/pull", methods=["GET"])
@login_required
def pull_server_changes():
    """
    Returns all changes made on the server since the device's last sync timestamp.
    """
    last_sync = request.args.get("since")
    return jsonify(
        {
            "status": "success",
            "message": "Sync pull endpoint active.",
            "since": last_sync,
            "updates": [],
        }
    )


@bp.route("/resolve", methods=["POST"])
@login_required
def resolve_conflict():
    """
    Allows a user to manually resolve a data collision.
    """
    _data = request.get_json() or {}
    return (
        jsonify({"status": "success", "message": "Conflict resolution stub active."}),
        200,
    )
