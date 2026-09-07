from flask import Blueprint

bp = Blueprint("offline_sync", __name__, url_prefix="/sync")

from . import routes  # noqa: E402, F401
