from flask import Blueprint

from . import routes  # noqa: F401

bp = Blueprint("referrals", __name__, url_prefix="/referrals")
