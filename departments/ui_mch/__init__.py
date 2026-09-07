from flask import Blueprint
bp = Blueprint("ui_mch", __name__, url_prefix="/ui/mch", template_folder="templates")
from . import routes  # noqa: E402, F401
