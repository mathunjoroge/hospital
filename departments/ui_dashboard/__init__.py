from flask import Blueprint

bp = Blueprint("ui_dashboard", __name__, url_prefix="/ui", template_folder="templates")

from . import routes  # noqa: E402, F401
