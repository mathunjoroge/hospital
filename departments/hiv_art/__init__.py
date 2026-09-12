from flask import Blueprint

bp = Blueprint("hiv_art", __name__, url_prefix="/hiv_art")

from . import routes  # noqa: F401
