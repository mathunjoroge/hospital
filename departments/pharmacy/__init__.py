from flask import Blueprint

bp = Blueprint("pharmacy", __name__, template_folder="templates")

from . import (  # noqa: F401    ai_discovery,
    dispensing,
    inventory,
    reports,
    stock_ops,
)
