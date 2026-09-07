from flask import Blueprint
bp = Blueprint("ui_billing", __name__, url_prefix="/ui/billing", template_folder="templates")
from . import routes  # noqa: E402, F401
