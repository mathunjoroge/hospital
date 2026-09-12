from flask import Blueprint

bp = Blueprint("tb_dots", __name__, url_prefix="/tb_dots")

from . import routes  # noqa: F401
