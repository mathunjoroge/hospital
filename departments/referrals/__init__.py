from flask import Blueprint

bp = Blueprint("referrals", __name__, url_prefix="/referrals")

from . import routes  # noqa: E402, F401
