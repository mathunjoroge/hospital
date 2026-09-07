from flask import Blueprint
bp = Blueprint("ui_referrals", __name__, url_prefix="/ui/referrals", template_folder="templates")
from . import routes  # noqa: E402, F401
